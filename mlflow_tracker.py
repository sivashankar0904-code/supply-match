"""MLflow tracking for the supply-match pipeline.

Two entry points:
    log_embed_run()  — one run per index build (called after embed_data.py finishes)
    log_inference()  — per /match request; appended as time-series metrics to a
                       single long-lived "inference" run

Both are fail-safe: any MLflow error is logged and swallowed so tracking can
never break the pipeline or the API.
"""

import logging
import threading

import mlflow
from mlflow.tracking import MlflowClient

import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_setup_done = False
_inference_run_id: str | None = None
_inference_step = 0


def _ensure_setup() -> None:
    global _setup_done
    if not _setup_done:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        _setup_done = True


def log_embed_run(*, num_masters: int, dim: int, duration_s: float) -> None:
    """Log one MLflow run describing an index build. Call after embed_data writes artifacts."""
    try:
        _ensure_setup()
        with mlflow.start_run(run_name="embed", tags={"stage": "embed"}):
            mlflow.log_params(
                {
                    "model_name": config.MODEL_NAME,
                    "sample_size": "all" if config.SAMPLE_SIZE is None else config.SAMPLE_SIZE,
                    "batch_size": config.EMBED_BATCH_SIZE,
                    "embedding_dim": dim,
                }
            )
            mlflow.log_metrics(
                {
                    "num_masters": num_masters,
                    "embed_time_seconds": round(duration_s, 3),
                    "items_per_sec": round(num_masters / duration_s, 2) if duration_s else 0.0,
                }
            )
    except Exception as exc:  # tracking must never break the build
        log.warning("MLflow log_embed_run failed: %s", exc)


def log_inference(*, num_queries: int, k: int, latency_ms: float, top_score: float) -> None:
    """Append one /match request as a metric step on the shared inference run."""
    global _inference_run_id, _inference_step
    try:
        _ensure_setup()
        client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
        with _lock:
            if _inference_run_id is None:
                exp = mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
                run = client.create_run(
                    exp.experiment_id,
                    run_name="inference",
                    tags={"stage": "inference", "mlflow.runName": "inference"},
                )
                _inference_run_id = run.info.run_id
                client.log_param(_inference_run_id, "model_name", config.MODEL_NAME)
            run_id = _inference_run_id
            step = _inference_step
            _inference_step += 1
        client.log_metric(run_id, "latency_ms", round(latency_ms, 2), step=step)
        client.log_metric(run_id, "num_queries", num_queries, step=step)
        client.log_metric(run_id, "k", k, step=step)
        client.log_metric(run_id, "top_score", round(top_score, 4), step=step)
    except Exception as exc:  # tracking must never break a request
        log.warning("MLflow log_inference failed: %s", exc)
