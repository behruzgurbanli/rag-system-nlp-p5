# Improvement Notes

## Completed in preprocessing round

- Added deterministic cleanup rules for common PDF word-joining artifacts.
- Rebuilt `data/processed`, `data/chunks`, and `data/vector_store`.
- Chunking remains valid after cleanup:
  - 58 documents
  - 58 courses
  - 336 chunks
  - min chunk size: 206 tokens
  - max chunk size: 486 tokens
  - 0 chunks under 200
  - 0 chunks over 500
- Retrieval-only evaluation remains stable:
  - Top-1 course accuracy: 8/8
  - Top-3 course recall: 8/8

## Quality impact

- High-impact joined artifacts in the inspected list dropped from 139 matches
  to 1 match.
- The ANT 101 grading chunk is now much more readable, including phrases like
  "provide grades", "feedback within", "keep track", "please email",
  "change grades", and "meets basic standards".

## Possible next improvements

- Evaluation questions expanded from 8 to 14.
- Retrieval-only evaluation on the expanded set:
  - Top-1 course accuracy: 14/14
  - Top-3 course recall: 14/14
- Added additional targeted cleanup rules found from expanded evaluation
  contexts, including `coursestudies`, `dropcourses`, `milestoneswithin`,
  and `willapprove`.
- Current chunking after all preprocessing improvements:
  - 336 chunks
  - min chunk size: 206 tokens
  - max chunk size: 486 tokens
  - 0 chunks under 200
  - 0 chunks over 500

## Next improvement candidates

- Run Gemini RAG-only evaluation after API quota cools down:
  `python -m src.rag.evaluate --provider gemini --baseline-provider fallback --model gemini-2.5-flash --top-k 3 --delay-seconds 15`
- Inspect `results/evaluation_results_gemini_rag_only.json` after running it
  from a terminal that has `GEMINI_API_KEY` set. The sandbox run cannot use the
  key exported in the user's shell.
- Improve remaining long-tail PDF joins only if they appear in retrieved
  contexts or generated answers.
- Consider a keyword boost for queries without explicit course codes.
- Test Streamlit UI manually from a terminal.

## Completed in implementation improvement round

- Added a harder retrieval evaluation set in
  `results/evaluation_questions_hard.json`.
  - These questions intentionally omit explicit course codes.
  - Current result with `top_k=5`: top-1 course accuracy is 5/6 and top-5
    course recall is 6/6.
  - The one top-1 miss is an ambiguous chemistry question where CHEM 101L lab
    ranks first and CHEM 101 lecture still appears in the retrieved set.
- Added a small paid/free-quota friendly baseline subset in
  `results/evaluation_questions_baseline_subset.json`.
  - It has 3 questions.
  - Running it with Gemini for both RAG and baseline costs 6 Gemini requests.
- Improved no-course-code retrieval behavior by widening the FAISS candidate
  pool before hybrid reranking.
- Re-ran the expanded retrieval-only evaluation:
  - Top-1 course accuracy: 14/14
  - Top-3 course recall: 14/14
- Updated the Streamlit UI into an interactive dashboard:
  - custom RAG question answering
  - custom retriever-only testing with no LLM quota cost
  - evaluation result browsing
  - corpus/chunk inspection
- Set the UI's default provider to fallback so opening the app or testing
  locally does not spend Gemini requests unless Gemini is selected manually.
- Removed Streamlit `use_container_width` calls to avoid the upcoming
  deprecation warning.

## Recommended next work after a break

- Run the 3-question Gemini baseline subset when the Gemini daily quota has
  enough remaining requests.
- Inspect the Gemini RAG-only answers in
  `results/evaluation_results_gemini_rag_only.json` and mark any answer that
  is too short or too extractive.
- If time allows, add a small manual answer-quality table for the report later:
  RAG answer correct/partial/wrong vs. baseline correct/partial/wrong.
- Keep the hard no-code set as a stress test, but do not overfit it unless the
  professor specifically tests ambiguous queries without course codes.

## Completed in answer-quality and UI round

- Reworked generator prompting so RAG and baseline use different instructions.
  - RAG now explicitly answers only from retrieved syllabus context.
  - Baseline now behaves like a true no-retrieval LLM baseline instead of
    always defaulting to "not enough information."
- Added retry logic for Gemini generation failures to reduce one-off 503 errors.
- Added `results/manual_answer_review.json` for a small manual quality review
  of the baseline subset.
- Added `results/demo_showcase.json` so the UI can present saved demo-ready
  examples from earlier good runs.
- Upgraded the Streamlit app with:
  - side-by-side RAG vs baseline answer display
  - manual answer-review section
  - curated demo showcase tab
