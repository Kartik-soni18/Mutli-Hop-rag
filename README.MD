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
