from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    # TODO: per-request top_k override isn't wired into the retriever yet.
    top_k: Optional[int] = None


class Source(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    embed_model: str
    index_loaded: bool
    num_vectors: int
