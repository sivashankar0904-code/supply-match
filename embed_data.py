"""Step 1 — embed master items and build the FAISS search index.

Reads entity_pairs.csv, dedupes to unique master items, embeds their names, and
writes a FAISS index plus a row-aligned metadata table to artifacts/.

Runs on the 10k sample by default; `SM_SAMPLE_SIZE=0` builds the full index with
no code changes.

Roadmap:
    Embed master items                ← this script
    Build the search/match API
    Test with sample queries
    Add MLflow tracking
    Add drift monitoring with Evidently
"""

import faiss
import pandas as pd

import config
from embedding import embed_texts


def load_master_items() -> pd.DataFrame:
    """Unique master items (one row per master_id), optionally truncated to the sample size."""
    df = pd.read_csv(config.ENTITY_PAIRS_CSV, usecols=["master_id", "master_name"])
    masters = df.drop_duplicates(subset="master_id").reset_index(drop=True)
    if config.SAMPLE_SIZE is not None:
        masters = masters.head(config.SAMPLE_SIZE)
    return masters


def build_index(masters: pd.DataFrame) -> faiss.Index:
    """Embed master names and pack them into a cosine-similarity (inner-product) index."""
    vecs = embed_texts(masters["master_name"].tolist(), show_progress=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    masters = load_master_items()
    print(f"Embedding {len(masters):,} master items with {config.MODEL_NAME} ...")

    index = build_index(masters)

    faiss.write_index(index, str(config.INDEX_PATH))
    masters.to_parquet(config.MASTER_META_PATH, index=False)

    print(f"  index   -> {config.INDEX_PATH} ({index.ntotal:,} vectors, dim {index.d})")
    print(f"  meta    -> {config.MASTER_META_PATH}")


if __name__ == "__main__":
    main()
