"""Embedding model wrapper."""

from __future__ import annotations

import numpy as np

from .config import EMBEDDING_MODEL


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer with normalized embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers is required. Install requirements.txt in the conda env."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")
