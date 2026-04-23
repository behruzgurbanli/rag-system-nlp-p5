"""Extract and lightly clean syllabus PDFs.

This script is the reproducible first stage of the project data pipeline.
It reads PDFs from data/raw_pdfs and writes extracted text files to
data/cleaned.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize common PDF extraction artifacts without changing meaning."""
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u00b7": "-",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([.!?,:;])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text with pdfplumber, which handles syllabi layouts well."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise SystemExit(
            "pdfplumber is required for PDF extraction. Install requirements.txt first."
        ) from exc

    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


def process_file(pdf_path: Path, output_dir: Path) -> dict[str, object]:
    raw_text = extract_pdf_text(pdf_path)
    cleaned_text = clean_text(raw_text)
    output_path = output_dir / f"{pdf_path.stem}_cleaned.txt"
    output_path.write_text(cleaned_text, encoding="utf-8")
    return {
        "source": str(pdf_path),
        "output": str(output_path),
        "words": len(cleaned_text.split()),
        "characters": len(cleaned_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and clean syllabus PDFs.")
    parser.add_argument("--input-dir", default="data/raw_pdfs")
    parser.add_argument("--output-dir", default="data/cleaned")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in {input_dir}")

    print(f"Found {len(pdf_files)} PDFs in {input_dir}")
    for pdf_path in pdf_files:
        result = process_file(pdf_path, output_dir)
        print(f"{pdf_path.name}: {result['words']} words")


if __name__ == "__main__":
    main()
