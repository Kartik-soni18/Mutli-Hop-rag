# MultiHop-RAG Solution

This repository contains my retrieval-augmented generation (RAG) solution for the [MultiHop-RAG dataset](https://github.com/yixuantt/MultiHop-RAG), a benchmark for questions that require retrieving and reasoning across multiple documents.

The project uses Sentence Transformers for embeddings, Qdrant for vector retrieval, FastAPI for serving, and Groq or Cerebras Cloud for LLM inference.

## Retrieval branches and publication dates

The planner returns a nonempty `branches` array. Each branch has its own
`retrieval_query` and optional `source`, `authors`, `published_from`,
`published_to`, `url`, and `url_prefix`. Use one source per branch, or omit it
for a source-free search. Results are merged and deduplicated, then reranked
against the original question (10 candidates per branch, 3 final chunks).
Evaluation CSV output stores the complete plan as JSON in `retrieval_branches`.

Publication filters accept only `YYYY-MM-DD`. Both bounds are inclusive:
`[2024-02-29, 2024-02-29]` matches every article on February 29. Either bound
can be omitted. Invalid dates and reversed ranges are rejected.
Indexed `published_at` contains the calendar date from the corpus, and an
internal `published_date_ordinal` supports numeric range matching. The ordinal
is excluded from answer context. The source corpus remains unchanged.

## Rebuild the index

Chunk size is 400 characters with a target overlap of 50 characters. Existing
indexes must be rebuilt to apply the chunking and date metadata changes.
Stop any process using the local Qdrant database, then run from the repo root:

```sh
.venv/bin/python -m src.rag.index
```

This recreates the local collection from `data/corpus.json`. Restart any
long-running agent or service afterward so cached retrieval services reopen
it. Install test dependencies with `.venv/bin/python -m pip install -e ".[dev]"`.
Run focused checks with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```
