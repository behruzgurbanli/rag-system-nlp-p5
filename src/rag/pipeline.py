"""End-to-end RAG and baseline pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_GEMINI_MODEL, DEFAULT_PROVIDER, DEFAULT_TOP_K, RESULTS_DIR, VECTOR_STORE_DIR
from .generator import build_context, generate_answer
from .retriever import Retriever, detect_course_code


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_GEMINI_MODEL,
    provider: str = DEFAULT_PROVIDER,
    baseline_provider: str | None = None,
    store_dir: Path = VECTOR_STORE_DIR,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    retriever = retriever or Retriever(store_dir)
    course_filter = detect_course_code(question)
    retrieved_chunks = retriever.search(question, top_k=top_k, course_filter=course_filter)
    context = build_context(retrieved_chunks)
    baseline_provider = baseline_provider or provider
    rag_answer = generate_answer(question=question, context=context, provider=provider, model=model)
    baseline_answer = generate_answer(
        question=question,
        context=None,
        provider=baseline_provider,
        model=model,
    )

    return {
        "question": question,
        "top_k": top_k,
        "course_filter": course_filter,
        "provider": provider,
        "baseline_provider": baseline_provider,
        "model": model,
        "rag_answer": rag_answer,
        "baseline_answer": baseline_answer,
        "retrieved_chunks": retrieved_chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question using the RAG pipeline.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["gemini", "openai", "fallback", "extractive"],
    )
    parser.add_argument(
        "--baseline-provider",
        default=None,
        choices=["gemini", "openai", "fallback", "extractive"],
    )
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = answer_question(
        args.question,
        top_k=args.top_k,
        provider=args.provider,
        baseline_provider=args.baseline_provider,
        model=args.model,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
