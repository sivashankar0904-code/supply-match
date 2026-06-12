"""Central config for the supply-match pipeline.

Every knob that differs between the 10k bring-up and the full 1.2M run lives
here and is overridable via env var, so scaling up is a config change (e.g.
`SM_SAMPLE_SIZE=0`) rather than a code change.
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

ENTITY_PAIRS_CSV = PROCESSED_DIR / "entity_pairs.csv"
INDEX_PATH = ARTIFACTS_DIR / "master.faiss"
MASTER_META_PATH = ARTIFACTS_DIR / "master_meta.parquet"

LOGS_DIR = ROOT / "logs"
INFERENCE_LOG_CSV = LOGS_DIR / "inference_log.csv"
DRIFT_REPORT_HTML = ROOT / "monitoring" / "drift_report.html"

# Inline drift check: every Nth request, average the recent `top_score`s and
# alert if mean match confidence has fallen below the threshold.
DRIFT_CHECK_EVERY = int(os.getenv("SM_DRIFT_CHECK_EVERY", "5"))
DRIFT_CHECK_WINDOW = int(os.getenv("SM_DRIFT_CHECK_WINDOW", "50"))
DRIFT_SCORE_THRESHOLD = float(os.getenv("SM_DRIFT_SCORE_THRESHOLD", "0.5"))

# Embedding model. all-MiniLM-L6-v2 is symmetric (no query/passage prefixes),
# 384-dim, and fast — a good default for short product-name matching. Swap to
# e.g. BAAI/bge-small-en-v1.5 for higher quality once the pipeline is proven.
MODEL_NAME = os.getenv("SM_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH_SIZE = int(os.getenv("SM_BATCH_SIZE", "256"))

# Number of unique master items to embed. Default 10k for bring-up; set
# SM_SAMPLE_SIZE=0 (or "all"/"none") to embed the full dataset.
_sample = os.getenv("SM_SAMPLE_SIZE", "10000").strip().lower()
SAMPLE_SIZE: int | None = None if _sample in ("", "0", "none", "all") else int(_sample)

# MLflow tracking. Default store is ./mlruns in the project root (local file
# store), which is what `mlflow ui` reads by default when run from here.
MLFLOW_TRACKING_URI = os.getenv("SM_MLFLOW_URI", (ROOT / "mlruns").as_uri())
MLFLOW_EXPERIMENT = os.getenv("SM_MLFLOW_EXPERIMENT", "supply-match")
