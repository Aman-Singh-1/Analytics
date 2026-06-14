# Evaluation queries

Ten queries run against the live pipeline (Groq `llama-3.3-70b-versatile`,
bge-small embeddings, cross-encoder rerank). The same queries run end-to-end in
[`eval.ipynb`](eval.ipynb), which calls the real API and captures the responses
shown there — this file is the readable summary.

Important context for reading these results: this demo ships a **small 150-doc
index** (the top-150 highest-scored questions) so it builds and deploys fast.
Those top questions are overwhelmingly *language* questions, so library topics
like pandas are barely covered. The full 50k-pair corpus (built by the same
code with `INDEX_MAX_DOCS=0`) closes most of these gaps. I've kept the
small-index failures in here on purpose — they show the system refusing
honestly instead of bluffing.

`rel` = top cosine relevance score; the refusal gate is at 0.66. A refusal can
happen in **two** places: at the gate (`no (gate)`, before the LLM is called) or
in the LLM itself when the retrieved context doesn't actually cover the question
(`no (LLM)`).

| # | Query | rel | Grounded? | Notes |
|---|-------|-----|-----------|-------|
| 1 | reverse a list | 0.79 | yes | Correct `[::-1]`. Sources are reverse-a-string etc.; model adapted them well. |
| 2 | difference between `is` and `==` | 0.88 | yes | Clean identity-vs-equality explanation with `id()` example. Best result of the set. |
| 3 | list comprehension vs map | 0.78 | yes | Balanced, runnable comparison of both forms. |
| 4 | decorators (one word) | 0.78 | yes | Ambiguous single word still retrieved the decorator-chain Q and answered usefully. |
| 5 | pandas: drop rows with NaN | 0.68 | no (LLM) | **Coverage gap.** Gate passes (just above 0.66), but the only pandas doc in the slice is about *renaming columns*, so the LLM declines rather than invent `dropna`. Works on the full index. |
| 6 | why do I get an IndentationError | 0.72 | no (LLM) | Gate passes but the retrieved context doesn't cover it, so the LLM refuses. Two-layer honesty. |
| 7 | deploy a Django app to AWS | 0.62 | no (gate) | Out-of-scope-ish; refused at the gate before any LLM call. |
| 8 | capital of France | 0.49 | no (gate) | Clearly non-Python. Refused at the gate as intended. |
| 9 | empty / whitespace question | - | n/a | API returns **422** (pydantic strips whitespace, then `min_length`), never reaches the pipeline. |
| 10 | repeat of #1 | - | yes | Served from the TTL cache: ~1 ms wall vs ~900 ms warm. |

## Selected transcripts

**#1 reverse a list** (grounded, 4 sources incl. 931092 reverse-a-string)
> ```python
> my_list = [1, 2, 3, 4, 5]
> reversed_list = my_list[::-1]
> print(reversed_list)  # [5, 4, 3, 2, 1]
> ```
> This uses the same extended slice syntax as reversing a string [1].

**#2 is vs ==** (grounded, sources 1504717, 1549801)
> The difference between `is` and `==` is that `is` checks identity (same object
> in memory) while `==` checks equality (same value) [1]. ... You can also use
> the `id()` function to check the memory address of an object.

**#5 pandas drop NaN** (gate passed at 0.68, LLM refused)
> I don't have a grounded answer for dropping rows with NaN in a pandas
> dataframe based on the provided context [1]-[4]. The context only covers
> renaming columns in pandas, connecting to a MySQL database, and removing an
> element from a list by index.

**#6 IndentationError** (gate passed at 0.72, LLM refused)
> I don't have a grounded answer. The provided context [1]-[4] does not cover the
> question about IndentationError in Python.

**#8 capital of France** (refused at the gate, rel 0.49)
> I don't have a grounded answer for that in my Stack Overflow sources. Try
> rephrasing it as a Python question...

## What this tells me

- The grounding holds. Off-topic questions get refused at the gate (#7, #8), and
  even when a borderline question slips past the gate, the prompt makes the model
  decline rather than invent (#5, #6). That two-layer behaviour is the bit I'd
  want a reviewer to look at.
- The honest weakness is **coverage**, not retrieval quality. The cross-encoder
  orders the candidates well; there just aren't pandas/library docs in a
  150-question slice. #5 is the clearest case: the gate lets it through because
  *some* pandas content exists, but it's the wrong pandas content, so the LLM
  refuses. The fix is the index size knob, not the pipeline.
- The relevance gate at 0.66 is tuned to this small index. bge-small has a high
  cosine floor (off-topic still scores ~0.5-0.6), so the threshold sits in the
  gap above that. It should be re-tuned when the index size changes.
