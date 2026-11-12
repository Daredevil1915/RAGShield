# RAGShield

A research-to-application project for document question answering and RAG availability testing, built with FastAPI, Streamlit, ChromaDB, LangChain and Ollama.

- **Document Q&A:** upload text PDFs, ask follow-up questions, filter documents, and inspect document/page citations. Retrieval and lexical grounding checks reject some unsupported answers; these are heuristics, not a guarantee of correctness or prompt-injection protection.
- **Security Research:** compare clean and poisoned retrieval with BBO blockers or adversarial suffixes. Retrieval success is reported separately from answer jamming, experiment storage is isolated, and injected blockers are cleaned up.

## Run locally

Use Python 3.12 and install [Ollama](https://ollama.com/).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-app.txt
cp .env.example .env
ollama pull mistral
ollama pull nomic-embed-text
python scripts/start_render.py
```

Open http://localhost:10000. The launcher starts the API on loopback port 8000 and stops both processes if either fails. Alternatively run `docker compose up --build` and open http://localhost:8501; Compose also starts Ollama and downloads the models.

## Deploy on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Daredevil1915/RAGShield)

1. Create a Render **Blueprint** from this repository; it reads `render.yaml`.
2. Set `OLLAMA_BASE_URL` to a **hosted Ollama server reachable from Render**, with both `mistral` and `nomic-embed-text` installed. Your laptop's localhost will not work. This service does not start Ollama. For a server with bearer authentication, add `OLLAMA_API_KEY` in Render's environment settings.
3. Deploy and open the service URL. Read the generated `DEMO_PASSWORD` in Render's environment settings to sign in, or replace it with your own password there.
4. Check that the sidebar reports **Backend ready**, upload a PDF, ask a question, and inspect its page citation. In Security Research select **Prepare demo corpus**, then **Build experimental collection** before running a clean/poisoned comparison.

The UI binds to Render's `$PORT`; the document API stays on loopback. Render checks `/_stcore/health` for process health. The API `/health` reports model readiness separately: a running web service can still have a model configuration error. A hosted model service must support **both generation and embedding**, not only chat.

The Blueprint uses a free service for demonstrations. Uploaded indexes and experiment output disappear on restart, redeploy or idle spin-down. For persistent data, use a paid service with a disk mounted at `/opt/render/project/src/data`, keeping `APP_DB_DIR` under that path. See [Render free service limits](https://render.com/docs/free) and the [Blueprint reference](https://render.com/docs/blueprint-spec). Model hosting is separate and may have its own cost.

Signed-in visitors share one document library and experiment store. This is a single-user portfolio demo, not a multi-tenant service. Run one research experiment at a time.

## Research setup

Set `APP_RESEARCH_DEMO=1` for the shipped demonstration: 24 gold passages from the project's MS-MARCO subset and 40 distractors. This does **not** reproduce the 50,024-passage benchmark or its metrics. The Render Blueprint enables this mode; the sample is embedded only when you select Prepare demo corpus.

For the original benchmark leave that variable unset and supply your existing `db/chroma_db_msmarco` collection (`my_collection`) locally. Databases, the full corpus, private reports and generated experiment output are excluded. The small prebuilt control blocker fixture is included because the app and tests use it.

Fresh BBO generation uses the installed dependencies. Fresh gradient suffix generation additionally needs `torch`, `transformers` and `einops`, plus sufficient memory; use a capable local machine. Prebuilt blockers work without that stack. Changing models, corpus size or retrieval configuration changes experiment results.

## Verify

```bash
python -m pytest
# Real models, after configuring Ollama:
python -m evaluation.run_eval
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
