# MultiHop-RAG

<!-- evaluation-dashboard:start -->

![Evaluation progress](assets/evaluation.svg)

**Latest: 74/100 normalized exact matches** · Errors: 0 · Updated: 2026-09-07T14:23:50.874371+00:00

| Evaluated commit | Matches | Errors | UTC timestamp |
| :-- | --: | --: | :-- |
| `f3a76f0a` | 74/100 | 0 | 2026-09-07T14:23:50.874371+00:00 |
| `9bd4a9ac` | 14/100 | 27 | 2026-09-05T13:45:50.223226+00:00 |

[Latest answers](results.csv) · [Full history](evaluation/history.csv)

<!-- evaluation-dashboard:end -->

**LLM:** `inclusionai/ling-3.0-flash`

**Sources:** [MultiHop-RAG — Yixuan Tang & Yi Yang (2024)](https://arxiv.org/abs/2401.15391) · [AI Credits](https://aicredits.in)

```mermaid
flowchart TD
    Q[User query] --> T[Tool selection · AI Credits]
    T --> E[Local embedding]
    E --> V[Qdrant retrieval]
    V --> R[Local reranking]
    R --> A[Final answer · AI Credits]
    L[LangChain / OpenAI SDK] -.-> T
    L -.-> A
    O[Langfuse · one trace per request] -.-> T
    O -.-> E
    O -.-> V
    O -.-> R
    O -.-> A
```

![Langfuse trace](<assets/Screenshot 2026-09-07 at 1.52.41 PM.png>)
