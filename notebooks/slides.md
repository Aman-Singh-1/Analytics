# Slide deck outline (10 slides)

Content only — build the actual deck separately.

## 1. Title
Python Q&A Assistant — grounded answers to Python questions for data-science
learners, with citations, over a REST API.

## 2. Problem & approach
- Learners ask the same Python questions that are already answered well on Stack
  Overflow; raw LLMs answer confidently but can hallucinate APIs.
- Approach: RAG over a curated Stack Overflow Q&A corpus, so every answer is
  grounded in real sources and cites them — and refuses when it has no basis.

## 3. Architecture
question -> relevance gate -> FAISS top-k -> cross-encoder rerank -> grounding
prompt -> Groq llama-3.3-70b -> answer + cited sources.
- Local embeddings + local vector store = zero external infra beyond the LLM.

## 4. Data pipeline
- Source: Kaggle Stack Overflow Python dump (~2 GB, millions of rows).
- Pair each question with its highest-scored answer (accepted-flag proxy).
- Quality filter (score >= 5), cap to top ~50k pairs.
- Preserve code blocks as fenced markdown through cleaning + chunking — critical
  for a Python assistant. 87% of docs contain code.

## 5. Retrieval
- bge-small-en-v1.5 embeddings, normalized; query-side instruction prefix.
- Retrieve broad (top-8), rerank precise (top-4) with a cross-encoder.
- Broad-then-precise: the cross-encoder reads the full (query, passage) pair, so
  it orders results far better than cosine similarity alone.

## 6. Generation
- Grounding prompt: answer only from context, prefer runnable code, cite sources
  by number, say "I don't know" when context doesn't cover it.
- Two-layer honesty: a relevance gate refuses off-topic queries before the LLM,
  and the prompt makes the LLM decline if retrieval was weak.

## 7. API
- FastAPI, async `/ask` and a real `/health` (reports index status + vector
  count for platform health checks).
- Index + models loaded once on startup (lifespan), never per request.
- TTL cache on the normalised question; 422 on empty input via pydantic.

## 8. Testing & eval
- pytest: API happy path, validation (422), cache hit, refusal; LLM mocked, one
  live integration test.
- 10 documented eval queries with transcripts. Honest failure case: pandas
  questions get refused on the small demo index (coverage gap, not a retrieval
  bug) — fixed by indexing the full corpus.

## 9. Scaling to 100+ concurrent users (the real numbers)
Measured locally: cold answer ~12s (model warmup), warm ~1-2s, **cache hit
~1ms**. Groq latency dominates the warm path; embedding + rerank are a few
hundred ms on CPU.

Levers, roughly in order of bang-for-buck:
- **Caching.** Repeat questions are common for learners. At even a 50% hit rate
  you halve LLM calls and those requests drop from ~1-2s to ~1ms. Move the
  in-memory TTL cache to Redis so it's shared across workers.
- **Async + workers.** `/ask` already offloads the blocking work to a thread, so
  the event loop stays free. Run gunicorn with N uvicorn workers (N ~= cores);
  each handles many concurrent awaits. This alone takes one box from ~1 RPS to
  tens.
- **Batch embedding** for the query path under load (group concurrent queries
  into one encode call).
- **Externalise state to scale horizontally.** FAISS in-process is single-node;
  move vectors to a managed store (Qdrant/Pinecone) and metadata to Postgres,
  then autoscale stateless API replicas behind a load balancer.
- **Cost levers.** Cache hit-rate is the biggest one (fewer LLM tokens). Beyond
  that: a smaller/cheaper LLM for easy questions, and reuse embeddings (they
  don't change between requests).

Back-of-envelope: with Redis caching at ~50% hit rate and 4 workers per 4-core
box, a single instance comfortably serves 100+ users whose traffic is mostly
repeated/overlapping questions; bursty unique traffic scales by adding replicas.

## 10. Tradeoffs & next steps
- Chose zero-infra simplicity (FAISS, local embeddings) over scale-out
  readiness — fine for a demo, swappable for prod.
- Demo ships a small index for speed; full corpus is one env var away.
- Next: full index + re-tuned gate, retrieval metrics (hit@k/MRR), query
  rewrite, Redis + workers.
