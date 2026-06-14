from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    top_k: int = 8
    rerank_top_n: int = 4
    use_reranker: bool = True

    # bge-small has a high cosine floor (off-topic still scores ~0.5-0.6), so the
    # refusal gate sits in the gap above that. tune per index/embedding model.
    sim_threshold: float = 0.66

    index_dir: str = "data/faiss_index"
    # cap docs that go into the index (0 = all). small caps build fast for demos.
    index_max_docs: int = 0

    # repeat questions are common in a learning context, so a short TTL cache
    # absorbs most of the duplicate load without going stale on index rebuilds
    cache_size: int = 512
    cache_ttl: int = 3600

    log_level: str = "INFO"


settings = Settings()
