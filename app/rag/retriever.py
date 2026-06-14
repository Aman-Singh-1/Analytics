"""Retrieve relevant chunks, optionally reranked.

Retrieve broad with the bi-encoder (top_k), then rerank precise with a
cross-encoder down to rerank_top_n. The cross-encoder sees the full
(query, passage) pair so it's far better at ordering than cosine alone --
this is the main quality lever in the pipeline.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.rag.index import load_index


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    base = load_index().as_retriever(search_kwargs={"k": settings.top_k})
    if not settings.use_reranker:
        return base

    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    encoder = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
    compressor = CrossEncoderReranker(model=encoder, top_n=settings.rerank_top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base
    )


def retrieve(question: str) -> list[Document]:
    return get_retriever().invoke(question)
