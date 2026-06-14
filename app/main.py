from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.logging import get_logger, setup_logging
from app.rag import pipeline
from app.rag.index import load_index
from app.rag.retriever import get_retriever

log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    store = load_index()
    app.state.num_vectors = store.index.ntotal

    # warm the retriever (loads reranker) and the LLM chain so the first real
    # request doesn't eat the cold-start cost
    get_retriever()
    pipeline._chain()
    log.info("startup: index loaded, %d vectors", app.state.num_vectors)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Python Q&A Assistant", lifespan=lifespan)
    # open CORS for the demo; lock this down to known origins in prod.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
