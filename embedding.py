"""Embedding helpers shared by the index builder and the search/match API.

Vectors are L2-normalized so that a FAISS inner-product index computes cosine
similarity directly.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load (and cache) the embedding model for the current process."""
    return SentenceTransformer(config.MODEL_NAME)


def embed_texts(texts: list[str], *, show_progress: bool = False) -> np.ndarray:
    """Embed a list of strings into an (n, dim) float32, L2-normalized array."""
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return np.ascontiguousarray(vecs, dtype=np.float32)
