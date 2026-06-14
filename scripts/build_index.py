"""Offline index build: corpus.parquet -> chunk -> embed -> persist FAISS.

Run once locally. The API loads the persisted index at startup and never
rebuilds it on a cold start.
"""

import time
from pathlib import Path

from app.core.config import settings
from app.rag.index import build_index


def _dir_size_mb(path: str) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def main() -> None:
    start = time.perf_counter()
    stats = build_index()
    elapsed = time.perf_counter() - start

    print(
        f"docs={stats['docs']:,} chunks={stats['chunks']:,} "
        f"size={_dir_size_mb(stats['index_dir']):.0f}MB "
        f"time={elapsed:.0f}s -> {settings.index_dir}"
    )


if __name__ == "__main__":
    main()
