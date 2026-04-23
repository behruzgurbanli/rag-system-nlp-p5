"""Interactive Streamlit UI for the syllabus RAG system."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config import CHUNKS_PATH, DEFAULT_GEMINI_MODEL, DEFAULT_TOP_K, VECTOR_STORE_DIR
from src.rag.pipeline import answer_question
from src.rag.retriever import Retriever, detect_course_code


RESULTS_DIR = PROJECT_ROOT / "results"


@st.cache_resource
def load_retriever() -> Retriever:
    return Retriever()


@st.cache_data
def load_chunks() -> list[dict[str, Any]]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_json(path: str) -> Any:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def find_result_by_id(path: str, result_id: str) -> dict[str, Any] | None:
    data = load_json(path)
    if not isinstance(data, list):
        return None
    for item in data:
        if item.get("id") == result_id:
            return item
    return None


def chunks_to_frame(chunks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for chunk in chunks:
        rows.append(
            {
                "rank": chunk.get("rank"),
                "score": chunk.get("score"),
                "course": chunk["course_code"],
                "source": chunk["source_file"],
                "chunk": chunk["chunk_index"] + 1,
                "tokens": chunk["token_count"],
                "preview": chunk["text"][:240],
            }
        )
    return pd.DataFrame(rows)


def render_retrieved_chunks(chunks: list[dict[str, Any]]) -> None:
    st.dataframe(chunks_to_frame(chunks), width="stretch", hide_index=True)

    chart_df = pd.DataFrame(
        {
            "rank": [f"#{chunk['rank']} {chunk['course_code']}" for chunk in chunks],
            "score": [chunk["score"] for chunk in chunks],
        }
    )
    st.bar_chart(chart_df, x="rank", y="score")

    for chunk in chunks:
        with st.expander(
            f"#{chunk['rank']} {chunk['course_code']} | "
            f"score={chunk['score']} | {chunk['source_file']}"
        ):
            st.write(chunk["text"])


def evaluation_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected": item["expected_course"],
                "course_filter": item["course_filter"],
                "retrieved": " | ".join(item["retrieved_courses"]),
                "top1": item["top1_course_correct"],
                "topk": item["expected_course_in_top_k"],
                "rag_answer": item["rag_answer"],
            }
        )
    return pd.DataFrame(rows)


def manual_review_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": item["id"],
                "question": item["question"],
                "expected_course": item["expected_course"],
                "rag_quality": item["rag_quality"],
                "baseline_quality": item["baseline_quality"],
                "comparison": item["comparison"],
                "status": item["status"],
            }
            for item in items
        ]
    )


st.set_page_config(page_title="Course Syllabi RAG", layout="wide")

st.title("Course Syllabi RAG System")
st.caption("Interactive retrieval-augmented question answering over university syllabi.")

with st.sidebar:
    st.header("Runtime")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=DEFAULT_TOP_K)
    provider = st.selectbox("Generator", ["fallback", "gemini", "openai"], index=0)
    baseline_provider = st.selectbox("Baseline generator", ["fallback", "gemini", "openai"], index=0)
    model = st.text_input("Model", value=DEFAULT_GEMINI_MODEL)
    if baseline_provider == "fallback":
        st.caption("Baseline is currently using fallback mode, not an API model.")

    st.divider()
    gemini_status = "configured" if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") else "not configured"
    openai_status = "configured" if os.getenv("OPENAI_API_KEY") else "not configured"
    st.write(f"Gemini key: {gemini_status}")
    st.write(f"OpenAI key: {openai_status}")

    st.divider()
    st.write(f"Vector store: `{VECTOR_STORE_DIR.name}`")

chunks = load_chunks()
retriever = load_retriever()

tab_ask, tab_retrieve, tab_eval, tab_demo, tab_corpus = st.tabs(
    ["Ask A Question", "Retriever Test", "Evaluation", "Demo Showcase", "Corpus"]
)

with tab_ask:
    st.subheader("Custom Question Test")
    st.write("Use this for professor-provided questions during the demo.")

    examples = [
        "What is the grading policy for ANT 101?",
        "Who is the instructor for PLS 140?",
        "What prerequisites are listed for CSCI 332?",
        "What are the assessment components for CHEM 101?",
        "What textbook or resources are used in MATH 274?",
    ]
    selected_example = st.selectbox("Example questions", examples)
    question = st.text_area("Question", value=selected_example, height=90)

    if st.button("Run RAG", type="primary") and question.strip():
        with st.spinner("Retrieving context and generating answer..."):
            result = answer_question(
                question.strip(),
                top_k=top_k,
                provider=provider,
                baseline_provider=baseline_provider,
                model=model,
                retriever=retriever,
            )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Detected course", result["course_filter"] or "None")
        col_b.metric("Chunks retrieved", len(result["retrieved_chunks"]))
        col_c.metric("Generator", result["provider"])
        st.caption(f"Baseline provider used for this run: {result['baseline_provider']}")

        st.subheader("RAG Answer")
        st.write(result["rag_answer"])

        st.subheader("Baseline Answer")
        st.write(result["baseline_answer"])

        st.subheader("Retrieved Evidence")
        render_retrieved_chunks(result["retrieved_chunks"])

with tab_retrieve:
    st.subheader("Retriever-Only Test")
    st.write("Use this when you want to test retrieval without spending LLM API requests.")

    retrieval_query = st.text_input("Retrieval query", value="attendance policy for WCS 150")
    manual_filter = st.text_input("Optional course filter", value="")

    if st.button("Search Chunks") and retrieval_query.strip():
        course_filter = manual_filter.strip() or detect_course_code(retrieval_query)
        results = retriever.search(retrieval_query.strip(), top_k=top_k, course_filter=course_filter)

        col_a, col_b = st.columns(2)
        col_a.metric("Course filter", course_filter or "None")
        col_b.metric("Results", len(results))
        render_retrieved_chunks(results)

with tab_eval:
    st.subheader("Evaluation Dashboard")

    eval_options = {
        "Gemini RAG-only": "results/evaluation_results_gemini_rag_only.json",
        "Retrieval only": "results/evaluation_results_retrieval_only.json",
        "Hard retrieval only": "results/evaluation_results_hard_retrieval_only.json",
        "Baseline subset": "results/evaluation_results_baseline_subset.json",
        "Latest default": "results/evaluation_results.json",
    }
    selected_eval = st.selectbox("Evaluation file", list(eval_options))
    eval_results = load_json(eval_options[selected_eval])

    if not eval_results:
        st.warning("No evaluation file found yet.")
    else:
        df = evaluation_frame(eval_results)
        total = len(df)
        top1 = int(df["top1"].sum())
        topk = int(df["topk"].sum())

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Questions", total)
        col_b.metric("Top-1 course accuracy", f"{top1}/{total}")
        col_c.metric("Top-k course recall", f"{topk}/{total}")

        acc_df = pd.DataFrame(
            {
                "metric": ["Top-1 accuracy", "Top-k recall"],
                "score": [top1 / total, topk / total],
            }
        )
        st.bar_chart(acc_df, x="metric", y="score")

        st.dataframe(
            df[["id", "question", "expected", "retrieved", "top1", "topk"]],
            width="stretch",
            hide_index=True,
        )

        selected_id = st.selectbox("Inspect answer", df["id"].tolist())
        row = next(item for item in eval_results if item["id"] == selected_id)
        st.write("Question:", row["question"])

        answer_col_a, answer_col_b = st.columns(2)
        with answer_col_a:
            st.markdown("**RAG Answer**")
            st.write(row["rag_answer"])
        with answer_col_b:
            st.markdown("**Baseline Answer**")
            st.write(row.get("baseline_answer", "No baseline answer saved in this file."))

        st.markdown("**Retrieved Chunks**")
        render_retrieved_chunks(row["retrieved_chunks"])

    manual_review = load_json("results/manual_answer_review.json")
    if manual_review:
        st.divider()
        st.subheader("Manual Answer Review")
        review_df = manual_review_frame(manual_review)
        st.dataframe(review_df, width="stretch", hide_index=True)

        selected_review_id = st.selectbox("Inspect reviewed item", review_df["id"].tolist())
        review_item = next(item for item in manual_review if item["id"] == selected_review_id)
        review_result = find_result_by_id(review_item["source_file"], review_item["id"])

        review_col_a, review_col_b, review_col_c = st.columns(3)
        review_col_a.metric("RAG quality", f"{review_item['rag_quality']}/10")
        review_col_b.metric("Baseline quality", f"{review_item['baseline_quality']}/10")
        review_col_c.metric("Comparison", review_item["comparison"])
        st.write(review_item["notes"])

        if review_result:
            answer_col_a, answer_col_b = st.columns(2)
            with answer_col_a:
                st.markdown("**Saved RAG Answer**")
                st.write(review_result["rag_answer"])
            with answer_col_b:
                st.markdown("**Saved Baseline Answer**")
                st.write(review_result["baseline_answer"])

with tab_demo:
    st.subheader("Demo Showcase")
    st.write("Curated saved examples for presentation and live discussion.")

    showcase_items = load_json("results/demo_showcase.json")
    if not showcase_items:
        st.warning("No demo showcase file found yet.")
    else:
        showcase_titles = [item["title"] for item in showcase_items]
        selected_title = st.selectbox("Showcase case", showcase_titles)
        selected_item = next(item for item in showcase_items if item["title"] == selected_title)
        showcase_result = find_result_by_id(selected_item["source_file"], selected_item["result_id"])

        st.info(selected_item["why_it_is_good"])
        if showcase_result:
            meta_col_a, meta_col_b, meta_col_c = st.columns(3)
            meta_col_a.metric("Expected course", showcase_result.get("expected_course", "N/A"))
            meta_col_b.metric("Top-1 correct", str(showcase_result.get("top1_course_correct", "N/A")))
            meta_col_c.metric("Top-k hit", str(showcase_result.get("expected_course_in_top_k", "N/A")))

            st.write("Question:", showcase_result["question"])

            answer_col_a, answer_col_b = st.columns(2)
            with answer_col_a:
                st.markdown("**RAG Answer**")
                st.write(showcase_result["rag_answer"])
            with answer_col_b:
                st.markdown("**Baseline Answer**")
                st.write(showcase_result.get("baseline_answer", "No baseline answer saved in this file."))

            st.markdown("**Retrieved Evidence**")
            render_retrieved_chunks(showcase_result["retrieved_chunks"])
        else:
            st.warning("Selected showcase result was not found in the saved evaluation file.")

with tab_corpus:
    st.subheader("Corpus And Chunk Statistics")

    chunk_df = pd.DataFrame(chunks)
    summary = load_json("data/chunks/chunking_summary.json")
    metadata = load_json("data/vector_store/metadata.json")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Documents", summary["stats"]["total_documents"] if summary else chunk_df["source_file"].nunique())
    col_b.metric("Courses", summary["stats"]["total_courses"] if summary else chunk_df["course_code"].nunique())
    col_c.metric("Chunks", len(chunk_df))
    col_d.metric("Embedding dim", metadata["embedding_dim"] if metadata else "N/A")

    size_df = chunk_df[["course_code", "token_count"]].rename(
        columns={"course_code": "course", "token_count": "tokens"}
    )
    st.write("Chunk token distribution")
    st.bar_chart(size_df.groupby("course")["tokens"].count().reset_index(name="chunks"), x="course", y="chunks")

    courses = ["All"] + sorted(chunk_df["course_code"].unique().tolist())
    selected_course = st.selectbox("Browse course chunks", courses)
    browser_df = chunk_df if selected_course == "All" else chunk_df[chunk_df["course_code"] == selected_course]
    browser_df = browser_df.assign(preview=browser_df["text"].str.slice(0, 260))
    st.dataframe(
        browser_df[["course_code", "source_file", "chunk_index", "token_count", "preview"]],
        width="stretch",
        hide_index=True,
    )
