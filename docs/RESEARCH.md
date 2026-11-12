# Research methodology and evaluation

[Back to RAGShield](../README.md)

## Experimental objective

RAGShield examines availability attacks against retrieval-augmented generation: whether an injected blocker document can suppress an answer that the clean system can produce.

An experiment distinguishes three observations:

1. **Valid target:** the clean system answers the question.
2. **Retrieval success:** the injected blocker reaches the retrieved context.
3. **Jamming success:** the poisoned system refuses or no longer conveys the correct answer, under the selected grading method.

Retrieval success alone is not evidence of jamming. Experiments run against a separate collection, and injected blockers are removed after evaluation, including when a run raises an error.

## Corpus and model scope

| Configuration | Corpus | Purpose |
| --- | --- | --- |
| Hosted demonstration | 24 gold passages and 40 distractors | Interactive workflow demonstration |
| Original research setup | 50,024 passages with the original model configuration | Benchmark reproduction with separately supplied research data |

Set `APP_RESEARCH_DEMO=1` for the shipped demonstration: 24 gold passages from the project's MS-MARCO subset and 40 distractors. This does **not** reproduce the 50,024-passage benchmark or its metrics. The Render Blueprint enables this mode; the sample is embedded only when you select Prepare demo corpus.

Cloud models and embedding endpoints get separate vector collections and demo database paths, so their vectors do not mix with
the original Ollama index or with a different cloud embedding model. Re-upload PDFs and prepare the demo corpus again after
changing the embedding model/provider. Retrieval thresholds also need recalibration when switching embedding models.

For the original benchmark use optional local mode (`MODEL_PROVIDER=ollama`, `EMBEDDING_PROVIDER=ollama`), leave
`APP_RESEARCH_DEMO` unset and supply your existing `db/chroma_db_msmarco` collection (`my_collection`). This optional mode
needs Ollama with `mistral` and `nomic-embed-text`; `docker compose up --build` starts that stack. It is not used by Render.
Databases, the full corpus, private reports and generated output are excluded. The small prebuilt control blocker fixture is included
because the app and tests use it. Cloud demo results are not a reproduction of the original Mistral/nomic benchmark.

Fresh BBO generation uses the installed dependencies. Fresh gradient suffix generation additionally needs `torch`, `transformers` and `einops`, plus sufficient memory; use a capable local machine. Prebuilt blockers work without that stack. Changing models, corpus size or retrieval configuration changes experiment results.

## Run checks and evaluation

```bash
python -m pytest
# Configured cloud or optional local models:
python -m evaluation.run_eval --pdf path/to/usenixsecurity25-shafran.pdf
```

Tests use deterministic fake models; PDF parsing, chunking, ChromaDB, API and UI run for real. They do not certify hosted model quality. GitHub Actions runs the tests on Linux/Python 3.12.

## Interpretation

Retrieval thresholds and lexical grounding checks are heuristics. They do not guarantee factual correctness or resistance to prompt injection. Results depend on the corpus, model, retrieval settings and grading method; cloud demo outcomes should not be presented as original benchmark results.
