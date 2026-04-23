"""Top-k retrieval over the syllabus vector store."""

from __future__ import annotations

import json
import pickle
import math
from pathlib import Path
from typing import Any

import numpy as np
import re

from .config import DEFAULT_TOP_K, VECTOR_STORE_DIR
from .embeddings import EmbeddingModel


def detect_course_code(query: str) -> str | None:
    match = re.search(r"\b([A-Za-z]{2,4})[-\s]?([0-9]{3,4}[A-Za-z]?)\b", query)
    if not match:
        return None
    return f"{match.group(1).upper()} {match.group(2).upper()}"


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "course", "for", "from",
    "how", "in", "is", "it", "listed", "of", "on", "or", "the", "to", "what",
    "when", "where", "which", "who", "with",
}


def tokenize_query(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 2}


class Retriever:
    def __init__(self, store_dir: Path = VECTOR_STORE_DIR):
        self.store_dir = store_dir
        self.metadata = json.loads((store_dir / "metadata.json").read_text(encoding="utf-8"))
        with (store_dir / "chunks.pkl").open("rb") as file:
            self.chunks: list[dict[str, Any]] = pickle.load(file)

        self.embeddings = np.load(store_dir / "embeddings.npy")
        self.embedder = EmbeddingModel(self.metadata["embedding_model"])
        self.faiss_index = self._load_faiss_index()
        self.chunk_terms = [tokenize_query(chunk["text"]) for chunk in self.chunks]

    def _load_faiss_index(self):
        index_path = self.store_dir / "faiss.index"
        if not index_path.exists():
            return None
        try:
            import faiss
        except ImportError:
            return None
        return faiss.read_index(str(index_path))

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        course_filter: str | None = None,
        hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embedder.encode([query], batch_size=1)
        course_filter = course_filter or detect_course_code(query)

        if self.faiss_index is not None:
            search_k = len(self.chunks) if course_filter else min(len(self.chunks), max(top_k * 10, 50))
            scores, indices = self.faiss_index.search(query_embedding, search_k)
            pairs = list(zip(indices[0].tolist(), scores[0].tolist(), strict=False))
        else:
            similarities = self.embeddings @ query_embedding[0]
            top_indices = np.argsort(-similarities)
            if not course_filter:
                top_indices = top_indices[: min(len(top_indices), max(top_k * 10, 50))]
            pairs = [(int(index), float(similarities[index])) for index in top_indices]

        if hybrid:
            query_terms = tokenize_query(query)
            reranked_pairs = []
            for index, score in pairs:
                lexical_overlap = len(query_terms & self.chunk_terms[index])
                lexical_boost = 0.04 * min(lexical_overlap, 5)
                reranked_pairs.append((index, score, score + lexical_boost))
            reranked_pairs.sort(key=lambda item: item[2], reverse=True)
            pairs = [(index, hybrid_score) for index, _score, hybrid_score in reranked_pairs]

        results = []
        for index, score in pairs:
            chunk = dict(self.chunks[index])
            if course_filter and chunk["course_code"] != course_filter:
                continue
            rank = len(results) + 1
            chunk["rank"] = rank
            chunk["score"] = round(float(score), 4)
            results.append(chunk)
            if len(results) == top_k:
                break
        return results
