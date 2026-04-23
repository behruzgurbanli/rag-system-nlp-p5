"""Vector index creation and search."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .config import CHUNKS_PATH, VECTOR_STORE_DIR
from .embeddings import EmbeddingModel


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_numpy_store(embeddings: np.ndarray, chunks: list[dict[str, Any]], output_dir: Path) -> None:
    np.save(output_dir / "embeddings.npy", embeddings)
    with (output_dir / "chunks.pkl").open("wb") as file:
        pickle.dump(chunks, file)


def build_faiss_index(embeddings: np.ndarray, output_dir: Path) -> bool:
    try:
        import faiss
    except ImportError:
        return False

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "faiss.index"))
    return True


def build_vector_store(
    chunks_path: Path = CHUNKS_PATH,
    output_dir: Path = VECTOR_STORE_DIR,
    model_name: str | None = None,
) -> dict[str, Any]:
    chunks = load_chunks(chunks_path)
    texts = [chunk["text"] for chunk in chunks]

    embedder = EmbeddingModel(model_name=model_name) if model_name else EmbeddingModel()
    embeddings = embedder.encode(texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_numpy_store(embeddings, chunks, output_dir)
    has_faiss = build_faiss_index(embeddings, output_dir)

    metadata = {
        "embedding_model": embedder.model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
        "faiss_index": has_faiss,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector store from chunks.")
    parser.add_argument("--chunks-path", default=str(CHUNKS_PATH))
    parser.add_argument("--output-dir", default=str(VECTOR_STORE_DIR))
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    metadata = build_vector_store(
        chunks_path=Path(args.chunks_path),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
