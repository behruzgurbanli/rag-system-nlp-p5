"""Answer generation for RAG and baseline modes."""

from __future__ import annotations

import os
import time

from .config import DEFAULT_GEMINI_MODEL, DEFAULT_GENERATION_MODEL, DEFAULT_PROVIDER


RAG_SYSTEM_PROMPT = (
    "You answer questions about university course syllabi. "
    "Use only the provided syllabus context. "
    "Do not invent details that are not supported by the context. "
    "Answer clearly and directly in 2-6 sentences or short bullet points, depending on the question. "
    "Prefer specific course facts such as instructors, grading weights, office hours, prerequisites, "
    "assessment components, textbooks, and policies when they are present. "
    "If the answer is not contained in the provided context, say that the syllabus context does not contain enough information."
)

BASELINE_SYSTEM_PROMPT = (
    "You are answering questions about university course syllabi without any retrieved syllabus context. "
    "Answer from general knowledge only if you can do so honestly. "
    "Do not pretend to know exact syllabus-specific details that were not provided. "
    "If the question requires course-specific facts such as instructor name, grading policy, attendance rules, "
    "office hours, or assigned textbook, say that you cannot verify the exact syllabus details without the course document."
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        header = (
            f"[{chunk['rank']}] Course: {chunk['course_code']} | "
            f"Source: {chunk['source_file']} | Score: {chunk['score']}"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def generate_with_openai(
    question: str,
    context: str | None = None,
    model: str = DEFAULT_GENERATION_MODEL,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return extractive_fallback(question, context)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai package is required for API generation.") from exc

    client = OpenAI(api_key=api_key)
    if context:
        instructions = RAG_SYSTEM_PROMPT
        user_prompt = (
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer using only the syllabus context."
        )
    else:
        instructions = BASELINE_SYSTEM_PROMPT
        user_prompt = (
            f"Question: {question}\n\n"
            "Answer without using retrieved context. If exact syllabus facts are unknown, say so."
        )

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_prompt,
            temperature=0.2,
            max_output_tokens=350,
        )
    except Exception as exc:
        fallback = extractive_fallback(question, context)
        return f"{fallback}\n\n[OpenAI generation failed: {type(exc).__name__}: {exc}]"

    return response.output_text.strip()


def generate_with_gemini(
    question: str,
    context: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return extractive_fallback(question, context)

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("google-genai package is required for Gemini generation.") from exc

    client = genai.Client(api_key=api_key)
    if context:
        instructions = RAG_SYSTEM_PROMPT
        prompt = (
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer using only the syllabus context."
        )
    else:
        instructions = BASELINE_SYSTEM_PROMPT
        prompt = (
            f"Question: {question}\n\n"
            "Answer without using retrieved context. If exact syllabus facts are unknown, say so."
        )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instructions,
                    temperature=0.2,
                    max_output_tokens=700,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = (response.text or "").strip()
            if text:
                return text
            last_error = RuntimeError("Empty Gemini response.")
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(4 * (attempt + 1))
                continue
            break

    fallback = extractive_fallback(question, context)
    return f"{fallback}\n\n[Gemini generation failed: {type(last_error).__name__}: {last_error}]"


def generate_answer(
    question: str,
    context: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
) -> str:
    provider = provider.lower().strip()
    if provider == "gemini":
        return generate_with_gemini(question, context, model or DEFAULT_GEMINI_MODEL)
    if provider == "openai":
        return generate_with_openai(question, context, model or DEFAULT_GENERATION_MODEL)
    if provider in {"fallback", "extractive"}:
        return extractive_fallback(question, context)
    raise ValueError(f"Unknown generator provider: {provider}")


def extractive_fallback(question: str, context: str | None = None) -> str:
    """Deterministic fallback used when no API key is configured."""
    if not context:
        return (
            "Baseline mode has no retrieval context. Configure GEMINI_API_KEY "
            "or OPENAI_API_KEY to generate a true LLM baseline answer."
        )

    lines = [line.strip() for line in context.splitlines() if line.strip()]
    content_lines = [line for line in lines if not line.startswith("[")]
    excerpt = " ".join(content_lines)[:900]
    return (
        "API key not configured, so this is an extractive fallback from the "
        f"retrieved syllabus context: {excerpt}"
    )
