"""Embedding model + FAISS build/load.

bge models are trained with an instruction on the *query* side only; passages
go in raw. We override embed_query to add it and leave embed_documents alone.
Embeddings are normalized so FAISS L2 distance ranks the same as cosine.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CORPUS_PATH = Path("data/processed/corpus.parquet")


class _BgeQueryEmbeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(BGE_QUERY_PREFIX + text)


def get_embeddings(show_progress: bool = False) -> HuggingFaceEmbeddings:
    cls = _BgeQueryEmbeddings if "bge" in settings.embed_model.lower() else HuggingFaceEmbeddings
    return cls(
        model_name=settings.embed_model,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        show_progress=show_progress,
    )


def _corpus_to_documents(df: pd.DataFrame) -> list[Document]:
    return [
        Document(
            page_content=row.text,
            metadata={
                "question_id": int(row.question_id),
                "title": row.title,
                "score": int(row.score),
                "url": row.url,
            },
        )
        for row in df.itertuples(index=False)
    ]


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(corpus_path: Path = CORPUS_PATH, index_dir: str | None = None) -> dict:
    index_dir = index_dir or settings.index_dir
    df = pd.read_parquet(corpus_path)
    if settings.index_max_docs:
        df = df.nlargest(settings.index_max_docs, "score")
    docs = _corpus_to_documents(df)
    chunks = chunk_documents(docs)

    embeddings = get_embeddings(show_progress=True)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(index_dir)

    return {"docs": len(docs), "chunks": len(chunks), "index_dir": index_dir}


@lru_cache(maxsize=1)
def load_index(index_dir: str | None = None) -> FAISS:
    index_dir = index_dir or settings.index_dir
    return FAISS.load_local(
        index_dir, get_embeddings(), allow_dangerous_deserialization=True
    )
