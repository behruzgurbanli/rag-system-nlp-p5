# Data Preparation

This folder contains the reproducible Part 1 pipeline.

1. `clean_pdfs.py` extracts and lightly normalizes text from `data/raw_pdfs`.
2. `preprocess_texts.py` repairs common PDF spacing artifacts and writes
   `data/processed`.
3. `chunk_documents.py` creates RAG-ready chunks in `data/chunks`.

The final chunk file for indexing is `data/chunks/chunks.json`.
