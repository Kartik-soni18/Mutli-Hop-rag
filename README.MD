# MultiHop-RAG Solution

This repository contains my retrieval-augmented generation (RAG) solution for the [MultiHop-RAG dataset](https://github.com/yixuantt/MultiHop-RAG), a benchmark for questions that require retrieving and reasoning across multiple documents.

The project uses Sentence Transformers for embeddings, Qdrant for vector retrieval, FastAPI for serving, and Groq or Cerebras Cloud for LLM inference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Add your API keys to `.env`, then build the index and start the server:

```bash
python -m src.rag.index
uvicorn src.api:app
```

The API exposes retrieval at `GET /query`, Groq generation at `POST /generate`, and Cerebras generation at `POST /generate/cerebras`.

`GET /query` accepts optional metadata filters. Repeating `doc_id`, `author`, or
`source` applies an `IN` filter; filters of different kinds are combined with
`AND`.

```text
/query?q=energy&source=Reuters&source=BBC
/query?q=election&title_text=presidential&published_from=2023-10-01T00:00:00Z
/query?q=technology&doc_id_min=10&doc_id_max=100
/query?q=markets&url_prefix=https://example.com/business/
```

The supported parameters are `doc_id`, `doc_id_min`, `doc_id_max`, `title`,
`title_text`, `author`, `source`, `published_from`, `published_to`, `url`, and
`url_prefix`. Publication timestamps must include a timezone.

## Evaluation

I currently evaluate a random sample of 30 questions from MultiHop-RAG, so this is
a partial test rather than a full-dataset benchmark. A seed can be supplied to the
sampling script with `--seed <number>` to select the same 30 questions again and
make runs reproducible.

Exact match (EM) is the scoring metric. The evaluator compares each generated answer
with its reference answer after trimming surrounding whitespace and converting both
values to uppercase. This makes the comparison case-insensitive while keeping every
other character exact.

The results come from `test/test_output.csv`. On every push, the GitHub Actions
workflow recalculates these metrics and updates this section when the results change.

<!-- exact-match-results:start -->
### Latest exact-match result

- **Exact match (EM): 10.0% (3/30)**
- Non-matching answers: 27
- Blank model answers: 2
- Most common reference answer: `YES` (12/30 questions)
- Majority-answer baseline: 40.0% (EM is 30.0 percentage points below it)

This is strict: surrounding whitespace is trimmed and both answers are
converted to uppercase. Punctuation, different Unicode characters, and
additional explanation still count as a mismatch, so the score highlights
both answer correctness and instruction-following.
<!-- exact-match-results:end -->
