"""Search/match core — turn variant names into ranked master-item candidates.

Loads the FAISS index and metadata once per process, then maps query strings to
top-k masters with cosine-similarity scores. Imported by the API; also runnable
directly for a quick CLI smoke test.
"""

from dataclasses import dataclass
from functools import lru_cache

import faiss
import pandas as pd

import config
from embedding import embed_texts


@dataclass(frozen=True)
class Match:
    master_id: str
    master_name: str
    score: float


@dataclass
class Matcher:
    index: faiss.Index
    meta: pd.DataFrame

    def match(self, queries: list[str], k: int = 5) -> list[list[Match]]:
        """Return the top-k master candidates for each query string."""
        if not queries:
            return []
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(embed_texts(queries), k)
        results: list[list[Match]] = []
        for row_scores, row_ids in zip(scores, ids):
            row: list[Match] = []
            for score, idx in zip(row_scores, row_ids):
                if idx == -1:  # faiss pads with -1 when fewer than k hits exist
                    continue
                rec = self.meta.iloc[idx]
                row.append(Match(str(rec.master_id), str(rec.master_name), float(score)))
            results.append(row)
        return results


@lru_cache(maxsize=1)
def get_matcher() -> Matcher:
    """Load (and cache) the index + metadata for the current process."""
    if not config.INDEX_PATH.exists() or not config.MASTER_META_PATH.exists():
        raise FileNotFoundError(
            f"Index artifacts not found at {config.ARTIFACTS_DIR}. Run `python embed_data.py` first."
        )
    index = faiss.read_index(str(config.INDEX_PATH))
    meta = pd.read_parquet(config.MASTER_META_PATH)
    return Matcher(index=index, meta=meta)


if __name__ == "__main__":
    matcher = get_matcher()
    for query, hits in zip(
        ["bathroom exhaust fan with light", "wireless bluetooth earbuds"],
        matcher.match(["bathroom exhaust fan with light", "wireless bluetooth earbuds"], k=3),
    ):
        print(f"Q: {query}")
        for hit in hits:
            print(f"  {hit.score:.3f}  {hit.master_id}  {hit.master_name[:70]}")
        print()
