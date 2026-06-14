# Evaluation queries

Ten queries run against the live pipeline (Groq `llama-3.3-70b-versatile`,
bge-small embeddings, cross-encoder rerank). Answers are trimmed; sources are
shown as Stack Overflow question IDs.

Important context for reading these results: this demo ships a **small 150-doc
index** (the top-150 highest-scored questions) so it builds and deploys fast.
Those top questions are overwhelmingly *language* questions, so library topics
like pandas are barely covered. The full 50k-pair corpus (built by the same
code with `INDEX_MAX_DOCS=0`) closes most of these gaps. I've kept the
small-index failures in here on purpose — they show the system refusing
honestly instead of bluffing.

`rel` = top cosine relevance score; the refusal gate is at 0.66.

| # | Query | rel | Grounded? | Notes |
|---|-------|-----|-----------|-------|
| 1 | reverse a list | 0.79 | yes | Correct `[::-1]`. Sources are reverse-a-string etc.; model adapted them well. |
| 2 | pandas: drop rows with NaN | 0.65 | no (gate) | **Failure case.** No pandas docs in the 150-doc index, so it refuses. Works on the full index. |
| 3 | difference between `is` and `==` | 0.88 | yes | Clean identity-vs-equality explanation with example. Best result of the set. |
| 4 | why do I get an IndentationError | 0.72 | no (LLM) | Gate passed but the *LLM* refused: retrieved context didn't actually cover it. Two-layer honesty. |
| 5 | list comprehension vs map | 0.78 | yes | Balanced, runnable comparison. |
| 6 | decorators (one word) | 0.79 | yes | Ambiguous single word still retrieved the decorator-chain Q and answered usefully. |
| 7 | deploy a Django app to AWS | 0.62 | no (gate) | Out-of-scope-ish; refused. Arguably could partial-answer with a bigger index. |
| 8 | capital of France | 0.49 | no (gate) | Clearly non-Python. Refused as intended. |
| 9 | empty / whitespace question | - | n/a | API returns **422** (pydantic `min_length`), never reaches the pipeline. |
| 10 | repeat of #1 | - | yes | Served from the TTL cache: ~1ms vs ~12s cold. |

## Selected transcripts

**#1 reverse a list** (grounded, sources 931092, 952914, 1207406, 36932)
> ```python
> my_list = [1, 2, 3, 4, 5]
> reversed_list = my_list[::-1]
> print(reversed_list)  # [5, 4, 3, 2, 1]
> ```
> This uses the same extended slice syntax as reversing a string [1].

**#3 is vs ==** (grounded, sources 1504717, 1549801, 9573244)
> The difference between `is` and `==` is that `is` checks identity (same object
> in memory) while `==` checks equality (same value) [1].

**#4 IndentationError** (gate passed at 0.72, model refused)
> I don't have a grounded answer. The provided context [1]-[4] doesn't cover the
> question about IndentationError in Python.

**#8 capital of France** (refused at the gate, rel 0.49)
> I don't have a grounded answer for that in my Stack Overflow sources. Try
> rephrasing it as a Python question...

## What this tells me

- The grounding holds. Off-topic questions get refused at the gate, and even
  when a borderline question slips past the gate, the prompt makes the model
  decline rather than invent (#4). That two-layer behaviour is the bit I'd want
  a reviewer to look at.
- The honest weakness is **coverage**, not retrieval quality. The cross-encoder
  orders the candidates well; there just aren't pandas/library docs in a
  150-question slice. The fix is the index size knob, not the pipeline.
- The relevance gate at 0.66 is tuned to this small index. bge-small has a high
  cosine floor (off-topic still scores ~0.5-0.6), so the threshold sits in the
  gap above that. It should be re-tuned when the index size changes.
