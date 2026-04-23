"""Shared configuration for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
RESULTS_DIR = PROJECT_ROOT / "results"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_GENERATION_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_PROVIDER = "gemini"
