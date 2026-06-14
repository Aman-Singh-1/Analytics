from __future__ import annotations

import asyncio

from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import AskRequest, AskResponse, HealthResponse
from app.rag import pipeline

router = APIRouter()
log = get_logger("api")

# repeated questions are common with learners, so cache on the normalised text.
_cache: TTLCache = TTLCache(maxsize=settings.cache_size, ttl=settings.cache_ttl)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    num = getattr(request.app.state, "num_vectors", 0)
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        embed_model=settings.embed_model,
        index_loaded=num > 0,
        num_vectors=num,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    key = req.question.lower().strip()

    cached = _cache.get(key)
    if cached is not None:
        log.info("ask cache=hit ms=%.0f q=%r", cached["latency_ms"], req.question[:60])
        return AskResponse(**cached)

    try:
        result = await asyncio.to_thread(pipeline.answer, req.question)
    except Exception:
        log.exception("ask failed q=%r", req.question[:60])
        raise HTTPException(status_code=500, detail="internal error")

    _cache[key] = result
    log.info(
        "ask cache=miss ms=%.0f grounded=%s sources=%d q=%r",
        result["latency_ms"],
        result["grounded"],
        len(result["sources"]),
        req.question[:60],
    )
    return AskResponse(**result)
