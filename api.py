"""Search/match API — FastAPI service over the FAISS master-item index.

Run:  uv run uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs

Endpoints:
    GET  /health          liveness + index size
    POST /match           one or many variant names -> top-k master candidates
"""

import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

import config
from mlflow_tracker import log_inference
from search import get_matcher


class MatchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, description="Variant/supplier item names to match.")
    k: int = Field(5, ge=1, le=100, description="Number of master candidates per query.")


class MatchHit(BaseModel):
    master_id: str
    master_name: str
    score: float


class QueryResult(BaseModel):
    query: str
    matches: list[MatchHit]


class MatchResponse(BaseModel):
    results: list[QueryResult]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model + index at startup so the first request isn't slow.
    get_matcher()
    yield


app = FastAPI(title="supply-match", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    matcher = get_matcher()
    return {"status": "ok", "model": config.MODEL_NAME, "indexed_masters": matcher.index.ntotal}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest, background_tasks: BackgroundTasks) -> MatchResponse:
    matcher = get_matcher()
    t0 = time.perf_counter()
    batched = matcher.match(req.queries, k=req.k)
    latency_ms = (time.perf_counter() - t0) * 1000

    results = [
        QueryResult(
            query=query,
            matches=[MatchHit(master_id=h.master_id, master_name=h.master_name, score=h.score) for h in hits],
        )
        for query, hits in zip(req.queries, batched)
    ]

    # Log after the response is sent so tracking I/O never adds request latency.
    top_score = max((h.score for hits in batched for h in hits), default=0.0)
    background_tasks.add_task(
        log_inference,
        num_queries=len(req.queries),
        k=req.k,
        latency_ms=latency_ms,
        top_score=top_score,
    )
    return MatchResponse(results=results)
