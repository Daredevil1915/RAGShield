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
3. Deploy and open the service URL. The app opens directly to the document workspace without a login screen.
4. Open **Service status** in the sidebar to check that the cloud APIs are configured, then upload a PDF,
   ask a question, and select **Explore the evidence** to inspect its source passages.
   In Security Research select **Prepare research workspace**, then **Run clean baseline** before running an experiment.

The UI binds to Render's `$PORT`; the document API stays on loopback. Render checks `/_stcore/health` for process health.
The API `/health` checks cloud configuration without making paid inference calls; `configured` does not certify credentials,
quota or live model access. Failed provider requests show an actionable error without exposing keys or raw provider responses.

The Blueprint uses a free service for demonstrations. Uploaded indexes and experiment output disappear on restart, redeploy or idle spin-down. For persistent data, use a paid service with a disk mounted at `/opt/render/project/src/data`, keeping `APP_DB_DIR` under that path. See [Render free service limits](https://render.com/docs/free) and the [Blueprint reference](https://render.com/docs/blueprint-spec). Model hosting is separate and may have its own cost.

Visitors share one document library and experiment store. Run one research experiment at a time.

## Embedding API errors during setup

The default configuration uses OpenAI for document embeddings. An HTTP 429 response can indicate either temporary throttling or an account quota restriction.

| Error | Action |
| --- | --- |
| Insufficient quota or exhausted credits | Check the embedding provider's API billing, credit balance and project limits. Repeated setup attempts do not restore quota. |
| Project or organization spending limit | Review the corresponding account limit before retrying. |
| Temporary rate limit | The app retries identified transient failures up to twice and respects short `Retry-After` delays. Longer delays are shown in the error message. |
| Unspecified HTTP 429 | Check both quota and rate limits; the status code alone does not identify the cause. |

After resolving provider access, select **Prepare research workspace** again. Completed corpus batches are retained and skipped on retry. The app does not automatically switch embedding models or mix vectors from different providers.

See the [OpenAI error-code reference](https://developers.openai.com/api/docs/guides/error-codes) for account and rate-limit distinctions.
