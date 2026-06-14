"""The RAG chain: gate on relevance, retrieve, ground the LLM, return sources.

The relevance gate is a real branch, not decoration: we score the question
against the index and refuse when nothing clears a cosine threshold, so
off-topic or non-Python questions get an honest "I don't know" instead of a
confident hallucination.
"""

from __future__ import annotations

import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.core.config import settings
from app.rag.index import load_index
from app.rag.prompts import ANSWER_PROMPT, REFUSAL
from app.rag.retriever import retrieve


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] {d.metadata['title']} ({d.metadata['url']})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def _dedup_sources(docs: list[Document]) -> list[dict]:
    seen, out = set(), []
    for d in docs:
        url = d.metadata["url"]
        if url not in seen:
            seen.add(url)
            out.append({"title": d.metadata["title"], "url": url})
    return out


@lru_cache(maxsize=1)
def _chain():
    llm = ChatGroq(
        model=settings.llm_model, temperature=0, api_key=settings.groq_api_key
    )
    return ANSWER_PROMPT | llm | StrOutputParser()


def _top_relevance(question: str) -> float:
    hits = load_index().similarity_search_with_score(question, k=settings.top_k)
    if not hits:
        return 0.0
    # FAISS returns squared L2 on normalized vectors, so cos = 1 - dist/2
    return max(1.0 - score / 2.0 for _, score in hits)


def answer(question: str) -> dict:
    start = time.perf_counter()

    if _top_relevance(question) < settings.sim_threshold:
        return {
            "answer": REFUSAL,
            "sources": [],
            "latency_ms": (time.perf_counter() - start) * 1000,
            "grounded": False,
        }

    docs = retrieve(question)
    text = _chain().invoke(
        {"context": _format_context(docs), "question": question}
    )
    return {
        "answer": text,
        "sources": _dedup_sources(docs),
        "latency_ms": (time.perf_counter() - start) * 1000,
        "grounded": True,
    }
