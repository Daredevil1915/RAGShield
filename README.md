# RAGShield

A research-to-application project for document question answering and RAG availability testing, built with FastAPI, Streamlit, ChromaDB, LangChain and Ollama.

- **Document Q&A:** upload text PDFs, ask follow-up questions, filter documents, and inspect document/page citations. Retrieval and lexical grounding checks reject some unsupported answers; these are heuristics, not a guarantee of correctness or prompt-injection protection.
- **Security Research:** compare clean and poisoned retrieval with BBO blockers or adversarial suffixes. Retrieval success is reported separately from answer jamming, experiment storage is isolated, and injected blockers are cleaned up.

## Run with cloud APIs

Use Python 3.12. No Ollama installation, model downloads or running laptop are required for the deployed demo.
The default setup calls [Ollama Cloud](https://docs.ollama.com/cloud) for chat (`gemma4:31b`) and a hosted
[embedding API](https://developers.openai.com/api/reference/resources/embeddings/methods/create) for PDF search
(`text-embedding-3-small`). Ollama's current cloud catalog lists chat models rather than embedding models,
so the demo uses two API keys. Account access, quotas and billing belong to each provider.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-app.txt
cp .env.example .env
# Fill in OLLAMA_API_KEY and OPENAI_API_KEY in .env before running.
python scripts/start_render.py
```

Open http://localhost:10000. The launcher starts the API on loopback port 8000 and stops both processes if either fails.
On Render these processes run on Render's servers; closing or switching off your computer does not stop them.
Document text and questions are sent to the configured cloud providers for inference.

For a single cloud provider/key, use `MODEL_PROVIDER=openai` with `OPENAI_API_KEY` for both chat and embeddings.
`OPENAI_BASE_URL`, `OPENAI_CHAT_MODEL` and `OPENAI_EMBED_MODEL` also support compatible hosted APIs.
If the chat provider does not serve embeddings, configure `OPENAI_EMBED_BASE_URL` and `OPENAI_EMBED_API_KEY`
for a separate embedding service. Compatible APIs must return OpenAI-style chat and float embedding responses.

## Deploy on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Daredevil1915/RAGShield)

1. Create a Render **Blueprint** from this repository; it reads `render.yaml`.
2. Set `OLLAMA_API_KEY` (from [Ollama account keys](https://ollama.com/settings/keys)) and `OPENAI_API_KEY`
   (for hosted embeddings) as Render environment secrets. The Blueprint sets `MODEL_PROVIDER=ollama_cloud`,
   `EMBEDDING_PROVIDER=openai` and the default models. No local Ollama server or tunnel is used.
3. Deploy and open the service URL. Read the generated `DEMO_PASSWORD` in Render's environment settings to sign in, or replace it with your own password there.
4. Check that the sidebar reports **Cloud APIs configured**, upload a PDF, ask a question, and inspect its page citation.
   This first request verifies provider authentication and model access. In Security Research select **Prepare demo corpus**,
   then **Build experimental collection** before running a clean/poisoned comparison.

The UI binds to Render's `$PORT`; the document API stays on loopback. Render checks `/_stcore/health` for process health.
The API `/health` checks cloud configuration without making paid inference calls; `configured` does not certify credentials,
quota or live model access. Failed provider requests show an actionable error without exposing keys or raw provider responses.

The Blueprint uses a free service for demonstrations. Uploaded indexes and experiment output disappear on restart, redeploy or idle spin-down. For persistent data, use a paid service with a disk mounted at `/opt/render/project/src/data`, keeping `APP_DB_DIR` under that path. See [Render free service limits](https://render.com/docs/free) and the [Blueprint reference](https://render.com/docs/blueprint-spec). Model hosting is separate and may have its own cost.

Signed-in visitors share one document library and experiment store. This is a single-user portfolio demo, not a multi-tenant service. Run one research experiment at a time.

## Research setup

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

## Verify

```bash
python -m pytest
# Configured cloud or optional local models:
python -m evaluation.run_eval --pdf path/to/usenixsecurity25-shafran.pdf
```

Tests use deterministic fake models; PDF parsing, chunking, ChromaDB, API and UI run for real. They do not certify hosted model quality. GitHub Actions runs the tests on Linux/Python 3.12.

## Source layout

```text
app/                     API, ingestion, retrieval, grounding, research orchestration
ui/                      Streamlit document and research views
scripts/start_render.py  API/UI process supervisor
suffix_attack/           Suffix research and jamming evaluation
bbo_attack.py            Black-box blocker generation
rag_utils.py             Shared Ollama and text processing helpers
examples/                Small demonstration corpus
evaluation/              Golden questions and evaluation runner
tests/                   Offline API, UI, document, RAG and research tests
render.yaml              Render deployment Blueprint
Dockerfile               App container
docker-compose.yml       Local Ollama, API and UI stack
```

Do not commit credentials, uploaded PDFs or generated databases.
