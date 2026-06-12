"""Drift monitoring for the search/match service.

1. record_inference()  — append production inference signals to logs/inference_log.csv,
                         one row per matched query (the data we later test for drift).
2. (next) run_drift_report() — compare a reference window vs the recent window with
                         Evidently to detect score/latency drift.

The CSV is the source of truth for drift analysis: each row is one query's match
outcome, so the distribution of `top_score` over time is what signals when incoming
supplier names have drifted away from what the master index covers.
"""

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

log = logging.getLogger(__name__)

FIELDS = ["timestamp", "query", "top_score", "n_matches", "k", "latency_ms"]
DRIFT_COLUMNS = ["top_score", "latency_ms"]
DEFAULT_WINDOW = 200
_lock = threading.Lock()

_request_counter = 0
_counter_lock = threading.Lock()


def record_inference(records: list[dict]) -> None:
    """Append one row per query to the inference log. `records` come from a single
    /match request; each dict has query, top_score, n_matches, k, latency_ms.

    A batch request expands to one row per query so the score distribution stays
    correct for drift analysis — a single-query request is therefore one row.
    """
    if not records:
        return
    try:
        ts = datetime.now(timezone.utc).isoformat()
        rows = [{"timestamp": ts, **r} for r in records]
        with _lock:
            config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            is_new = not config.INFERENCE_LOG_CSV.exists()
            with config.INFERENCE_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if is_new:
                    writer.writeheader()
                writer.writerows(rows)
    except Exception as exc:  # monitoring must never break a request
        log.warning("record_inference failed: %s", exc)


def maybe_check_drift() -> dict | None:
    """Increment the request counter; on every `DRIFT_CHECK_EVERY`-th request,
    average `top_score` over the last `DRIFT_CHECK_WINDOW` logged queries and warn
    if mean match confidence has fallen below `DRIFT_SCORE_THRESHOLD`.

    Returns the check result dict when a check runs, else None. Call once per
    request, after record_inference, so the latest rows are on disk.
    """
    global _request_counter
    with _counter_lock:
        _request_counter += 1
        count = _request_counter
    if count % config.DRIFT_CHECK_EVERY != 0:
        return None
    try:
        df = pd.read_csv(config.INFERENCE_LOG_CSV)
        recent = df.tail(config.DRIFT_CHECK_WINDOW)
        if recent.empty:
            return None
        avg = float(recent["top_score"].mean())
        alert = avg < config.DRIFT_SCORE_THRESHOLD
        result = {
            "n_requests": count,
            "window": len(recent),
            "avg_top_score": round(avg, 4),
            "threshold": config.DRIFT_SCORE_THRESHOLD,
            "alert": alert,
        }
        if alert:
            log.warning(
                "DRIFT ALERT: avg top_score %.3f over last %d queries < threshold %.2f "
                "(checked at request #%d)",
                avg,
                len(recent),
                config.DRIFT_SCORE_THRESHOLD,
                count,
            )
        else:
            log.info(
                "drift check ok: avg top_score %.3f over last %d queries (>= %.2f)",
                avg,
                len(recent),
                config.DRIFT_SCORE_THRESHOLD,
            )
        if config.DRIFT_AUTO_REPORT and len(df) >= 2:
            run_drift_report()
        return result
    except Exception as exc:  # monitoring must never break a request
        log.warning("maybe_check_drift failed: %s", exc)
        return None


def run_drift_report(
    window: int = DEFAULT_WINDOW,
    output_path: Path = config.DRIFT_REPORT_HTML,
) -> Path:
    """Compare the first `window` requests (baseline) against the last `window`
    (recent) and write an Evidently data-drift report to `output_path`.

    Drift on `top_score` is the signal that incoming supplier names no longer
    match the master index as well as they used to (e.g. new catalog, new vendor).
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    df = pd.read_csv(config.INFERENCE_LOG_CSV)
    if len(df) < 2:
        raise ValueError(f"Need at least 2 logged requests to compare; have {len(df)}.")
    if len(df) < 2 * window:
        log.warning(
            "Only %d rows for window=%d - baseline and recent windows overlap, "
            "so drift will be understated.",
            len(df),
            window,
        )

    baseline = df.head(window)[DRIFT_COLUMNS]
    recent = df.tail(window)[DRIFT_COLUMNS]

    data_def = DataDefinition(numerical_columns=DRIFT_COLUMNS)
    ref_ds = Dataset.from_pandas(baseline, data_definition=data_def)
    cur_ds = Dataset.from_pandas(recent, data_definition=data_def)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    print(
        f"Drift report -> {output_path}\n"
        f"  baseline: first {len(baseline)} requests | recent: last {len(recent)} requests\n"
        f"  columns : {', '.join(DRIFT_COLUMNS)}"
    )
    return output_path


if __name__ == "__main__":
    run_drift_report()
