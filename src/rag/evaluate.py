"""Run a lightweight RAG-vs-baseline evaluation set."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_GEMINI_MODEL, DEFAULT_PROVIDER, DEFAULT_TOP_K, RESULTS_DIR
from .pipeline import answer_question
from .retriever import Retriever


DEFAULT_QUESTIONS_PATH = RESULTS_DIR / "evaluation_questions.json"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "evaluation_results.json"
DEFAULT_OUTPUT_CSV = RESULTS_DIR / "evaluation_results.csv"


def load_questions(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_question(
    item: dict[str, str],
    top_k: int,
    model: str,
    provider: str,
    baseline_provider: str | None,
    retriever: Retriever,
) -> dict[str, Any]:
    result = answer_question(
        item["question"],
        top_k=top_k,
        model=model,
        provider=provider,
        baseline_provider=baseline_provider,
        retriever=retriever,
    )
    retrieved_courses = [chunk["course_code"] for chunk in result["retrieved_chunks"]]
    expected_course = item.get("expected_course")
    result["id"] = item.get("id")
    result["expected_course"] = expected_course
    result["retrieved_courses"] = retrieved_courses
    result["top1_course_correct"] = bool(retrieved_courses and retrieved_courses[0] == expected_course)
    result["expected_course_in_top_k"] = expected_course in retrieved_courses
    return result


def write_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "id",
        "question",
        "expected_course",
        "course_filter",
        "provider",
        "baseline_provider",
        "model",
        "retrieved_courses",
        "top1_course_correct",
        "expected_course_in_top_k",
        "rag_answer",
        "baseline_answer",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "id": result["id"],
                    "question": result["question"],
                    "expected_course": result["expected_course"],
                    "course_filter": result["course_filter"],
                    "provider": result["provider"],
                    "baseline_provider": result["baseline_provider"],
                    "model": result["model"],
                    "retrieved_courses": " | ".join(result["retrieved_courses"]),
                    "top1_course_correct": result["top1_course_correct"],
                    "expected_course_in_top_k": result["expected_course_in_top_k"],
                    "rag_answer": result["rag_answer"],
                    "baseline_answer": result["baseline_answer"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
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
        help="Use fallback here for fast RAG-only evaluation, or same provider for full baseline.",
    )
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0,
        help="Optional delay between questions to avoid free-tier LLM rate limits.",
    )
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    retriever = Retriever()
    results = []
    for index, item in enumerate(questions):
        if index and args.delay_seconds:
            time.sleep(args.delay_seconds)
        results.append(
            evaluate_question(
                item,
                args.top_k,
                args.model,
                args.provider,
                args.baseline_provider,
                retriever,
            )
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(results, Path(args.output_csv))

    top1 = sum(result["top1_course_correct"] for result in results)
    topk = sum(result["expected_course_in_top_k"] for result in results)
    print(f"Questions: {len(results)}")
    print(f"Top-1 course accuracy: {top1}/{len(results)}")
    print(f"Top-{args.top_k} course recall: {topk}/{len(results)}")
    print(f"Wrote {output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
