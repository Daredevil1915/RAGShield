# Deployment and configuration

[Back to RAGShield](../README.md)

## Required API keys

The default deployment uses Ollama Cloud for chat and OpenAI for embeddings. Enter the actual secret keys in the matching Render Value fields.

| Environment variable | Purpose | Key management |
| --- | --- | --- |
| `OLLAMA_API_KEY` | Ollama Cloud chat | [Ollama account keys](https://ollama.com/settings/keys) |
| `OPENAI_API_KEY` | Hosted document and query embeddings | [OpenAI API keys](https://platform.openai.com/api-keys) |

Store credentials in Render environment settings or a local `.env` file. Never commit them to the repository.

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
