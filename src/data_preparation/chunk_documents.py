"""Create reproducible RAG chunks from preprocessed syllabi.

The course requirement asks for 200-500 token chunks. For this project we use
whitespace tokens, which are stable, transparent, and close enough for the
course-level requirement. The default window is 400 tokens with 50-token overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


def extract_course_code(path: Path) -> str:
    name = path.stem.replace("_processed", "").replace("_cleaned", "")
    match = re.search(r"([A-Z]{2,4})[-_\s]?([0-9]{3,4}[A-Z]?)", name.upper())
    if not match:
        return name.replace("_", " ").upper()
    return f"{match.group(1)} {match.group(2)}"


def sliding_chunks(
    words: list[str],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[list[str]]:
    if not words:
        return []

    if len(words) <= chunk_size + min_chunk_size:
        return [words]

    step = chunk_size - overlap
    starts = [0]
    while starts[-1] + chunk_size < len(words):
        next_start = starts[-1] + step
        if len(words) - next_start < min_chunk_size:
            next_start = max(0, len(words) - chunk_size)
        if next_start <= starts[-1]:
            break
        starts.append(next_start)

    return [words[start : start + chunk_size] for start in starts]


def chunk_document(
    path: Path,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    words = text.split()
    course_code = extract_course_code(path)
    chunks = sliding_chunks(words, chunk_size, overlap, min_chunk_size)
    total_chunks = len(chunks)

    records = []
    for index, chunk_words in enumerate(chunks):
        chunk_text = " ".join(chunk_words)
        records.append(
            {
                "chunk_id": f"{course_code.replace(' ', '_')}_chunk_{index + 1:03d}",
                "course_code": course_code,
                "source_file": path.name,
                "chunk_index": index,
                "total_chunks": total_chunks,
                "token_count": len(chunk_words),
                "text": chunk_text,
            }
        )
    return records


def write_outputs(chunks: list[dict[str, Any]], output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "chunks.json"
    json_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "chunks.csv"
    fieldnames = [
        "chunk_id",
        "course_code",
        "source_file",
        "chunk_index",
        "total_chunks",
        "token_count",
        "text",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunks)

    counts = [chunk["token_count"] for chunk in chunks]
    courses = sorted({chunk["course_code"] for chunk in chunks})
    source_files = sorted({chunk["source_file"] for chunk in chunks})
    summary = {
        "config": config,
        "stats": {
            "total_documents": len(source_files),
            "total_courses": len(courses),
            "total_chunks": len(chunks),
            "total_tokens_in_chunks": sum(counts),
            "average_chunk_tokens": round(mean(counts), 2),
            "median_chunk_tokens": median(counts),
            "min_chunk_tokens": min(counts),
            "max_chunk_tokens": max(counts),
            "chunks_under_200": sum(count < 200 for count in counts),
            "chunks_200_to_500": sum(200 <= count <= 500 for count in counts),
            "chunks_over_500": sum(count > 500 for count in counts),
        },
    }
    summary_path = output_dir / "chunking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "CHUNKING REPORT",
        "===============",
        f"Total documents: {summary['stats']['total_documents']}",
        f"Total courses: {summary['stats']['total_courses']}",
        f"Total chunks: {summary['stats']['total_chunks']}",
        f"Average chunk tokens: {summary['stats']['average_chunk_tokens']}",
        f"Median chunk tokens: {summary['stats']['median_chunk_tokens']}",
        f"Min chunk tokens: {summary['stats']['min_chunk_tokens']}",
        f"Max chunk tokens: {summary['stats']['max_chunk_tokens']}",
        f"Chunks under 200: {summary['stats']['chunks_under_200']}",
        f"Chunks from 200 to 500: {summary['stats']['chunks_200_to_500']}",
        f"Chunks over 500: {summary['stats']['chunks_over_500']}",
    ]
    (output_dir / "chunking_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk processed syllabus files.")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/chunks")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--min-chunk-size", type=int, default=200)
    args = parser.parse_args()

    if args.overlap >= args.chunk_size:
        raise SystemExit("--overlap must be smaller than --chunk-size")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No processed text files found in {input_dir}")

    all_chunks: list[dict[str, Any]] = []
    for path in files:
        document_chunks = chunk_document(
            path=path,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_chunk_size=args.min_chunk_size,
        )
        all_chunks.extend(document_chunks)
        print(f"{path.name}: {len(document_chunks)} chunks")

    config = {
        "chunk_size_tokens": args.chunk_size,
        "overlap_tokens": args.overlap,
        "min_chunk_size_tokens": args.min_chunk_size,
        "strategy": "sliding_window_with_overlap",
    }
    write_outputs(all_chunks, output_dir, config)
    print(f"Wrote {len(all_chunks)} chunks to {output_dir}")


if __name__ == "__main__":
    main()
