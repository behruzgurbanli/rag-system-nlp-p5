# Course Syllabi RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for
answering questions about university course syllabi.

## Current Status

Part 1 is organized and reproducible:

- Domain: education / university course syllabi
- Raw dataset: 58 PDF syllabi
- Processed dataset: 58 text documents
- Corpus size: more than 100,000 words
- Chunking: 336 chunks, all between 200 and 500 whitespace tokens
- Vector store: FAISS index with 384-dimensional Sentence-BERT embeddings
- Evaluation: expanded retrieval set passes 14/14 top-1 and 14/14 top-3

## Project Structure

```text
data/
  raw_pdfs/       Original syllabus PDFs
  cleaned/        Text extracted from PDFs
  processed/      Cleaned text used for chunking
  chunks/         JSON/CSV chunks and chunking reports
  vector_store/   FAISS index files for Part 2
src/
  data_preparation/
    clean_pdfs.py
    preprocess_texts.py
    chunk_documents.py
  rag/            Retrieval, generation, and evaluation code for Part 2
app/              UI for final results
results/          Evaluation outputs
report/           Final report material
slides/           Presentation material
```

## Reproduce Part 1

The cleaned and processed files are already included. To rerun the full data
pipeline from PDFs:

```bash
python3 src/data_preparation/clean_pdfs.py
python3 src/data_preparation/preprocess_texts.py
python3 src/data_preparation/chunk_documents.py
```

If starting from the existing cleaned text files, run only:

```bash
python3 src/data_preparation/preprocess_texts.py
python3 src/data_preparation/chunk_documents.py
```

## Chunking Configuration

The default chunking configuration is:

- chunk size: 400 whitespace tokens
- overlap: 50 whitespace tokens
- minimum chunk size: 200 whitespace tokens
- strategy: sliding window with overlap

The generated summary is stored in `data/chunks/chunking_summary.json`.

## Implemented RAG Components

- Sentence-BERT embeddings: implemented
- FAISS vector database: implemented
- top-k retriever: implemented
- LLM answer generator: implemented with Gemini by default, OpenAI optional
- baseline comparison without retrieval: implemented
- evaluation examples and result tables: implemented in `results/`
- Streamlit UI: implemented in `app/`

## Reproduce Part 2

```bash
python -m src.rag.vector_store
python -m src.rag.pipeline "What is the grading policy for ANT 101?" --provider gemini --top-k 3
python -m src.rag.evaluate --provider gemini --top-k 3
```

For development runs that should not spend Gemini requests:

```bash
python -m src.rag.evaluate \
  --provider fallback \
  --baseline-provider fallback \
  --top-k 3 \
  --output-json results/evaluation_results_retrieval_only.json \
  --output-csv results/evaluation_results_retrieval_only.csv
```

For harder custom-query retrieval tests without explicit course codes:

```bash
python -m src.rag.evaluate \
  --questions results/evaluation_questions_hard.json \
  --provider fallback \
  --baseline-provider fallback \
  --top-k 5 \
  --output-json results/evaluation_results_hard_retrieval_only.json \
  --output-csv results/evaluation_results_hard_retrieval_only.csv
```

Run the UI:

```bash
streamlit run app/streamlit_app.py
```

By default, generation uses Gemini:

```bash
export GEMINI_API_KEY="your_key_here"
python -m src.rag.evaluate --provider gemini --model gemini-2.5-flash --top-k 3
```

OpenAI remains available as an optional provider:

```bash
export OPENAI_API_KEY="your_key_here"
python -m src.rag.evaluate --provider openai --model gpt-4.1-mini --top-k 3
```

Without an API key, the system uses an extractive fallback answer from retrieved
context.

For a small baseline comparison that uses fewer Gemini daily requests, run:

```bash
python -m src.rag.evaluate \
  --questions results/evaluation_questions_baseline_subset.json \
  --provider gemini \
  --baseline-provider gemini \
  --model gemini-2.5-flash \
  --top-k 3 \
  --delay-seconds 25 \
  --output-json results/evaluation_results_baseline_subset.json \
  --output-csv results/evaluation_results_baseline_subset.csv
```

This subset uses 6 Gemini requests because each of the 3 questions generates
one RAG answer and one no-retrieval baseline answer.
