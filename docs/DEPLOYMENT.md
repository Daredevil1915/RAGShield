# Deployment and configuration

[Back to RAGShield](../README.md)

## Public interview demo

The default hosted configuration prioritizes a usable visitor experience:

- A built-in three-page PDF is indexed automatically.
- Word-feature retrieval runs on Render with no embedding API, credits or model downloads.
- Ollama Cloud generates live answers when its configured key is available.
- If generation fails, document Q&A returns exact retrieved sentences with citations and a visible **Source-only answer** label.
- The 64-passage research corpus is prepared automatically. Live experiments use the cloud model; an explicitly recorded project run can be explored independently.

Word-feature retrieval uses vocabulary overlap. It is intentionally lighter than a semantic embedding model and may miss paraphrases. Research never replaces a failed live run with a fabricated outcome.

## Deploy on Render

[Deploy to Render](https://render.com/deploy?repo=https://github.com/Daredevil1915/RAGShield)

1. Create a Blueprint from this repository. Render reads `render.yaml`.
2. Enter `OLLAMA_API_KEY` from [Ollama account keys](https://ollama.com/settings/keys) to enable live generation. OpenAI credentials are not required for the public demo.
3. Open the service URL. There is no login or visitor configuration step.
4. Try a suggested question about the sample guide, upload a PDF, or open Security Research and run a clean baseline followed by a prebuilt-blocker experiment.

| Variable | Public demo value | Purpose |
| --- | --- | --- |
| `APP_PUBLIC_DEMO` | `1` | Enables no-key retrieval, the sample guide and source-only fallback |
| `MODEL_PROVIDER` | `ollama_cloud` | Uses Ollama's hosted API for live generation |
| `EMBEDDING_PROVIDER` | `hashing` | Deterministic word-feature vectors on Render |
| `APP_RESEARCH_DEMO` | `1` | Uses the small demonstration corpus |
| `APP_MIN_RELEVANCE` | `0.08` | Demo retrieval threshold |
| `OLLAMA_CLOUD_MODEL` | `gemma4:31b` | Configurable cloud chat model |

Public demo mode selects word-feature retrieval even if a previously deployed service still has `EMBEDDING_PROVIDER=openai`. This prevents an exhausted embedding account from breaking the visitor workflow. Set `APP_PUBLIC_DEMO=0` to honor advanced provider settings.

The launcher exposes Streamlit on Render's `$PORT` and keeps the API on loopback. `/_stcore/health` checks process health; `/health` reports configuration without paid model requests. A configured key does not certify live provider access.

Free Render hosting can take time to wake after inactivity and uses a temporary filesystem. Uploaded indexes and new results can disappear on restart; the built-in samples are recreated automatically. For durable uploads, use a persistent disk under the app's data path. Visitors share the demo library. Research mutation and evaluation calls are serialized within the service.

## Run on a development machine

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-app.txt
cp .env.example .env
python scripts/start_render.py
```

Open http://localhost:10000. Add `OLLAMA_API_KEY` in `.env` for live cloud generation. Your own computer is not needed once the app is deployed on Render. Store secrets in environment settings; never commit them.

## Optional semantic retrieval

For the full hosted embedding configuration, set:

```dotenv
APP_PUBLIC_DEMO=0
MODEL_PROVIDER=ollama_cloud
EMBEDDING_PROVIDER=openai
OLLAMA_API_KEY=your-ollama-key
OPENAI_API_KEY=your-embedding-key
OPENAI_EMBED_MODEL=text-embedding-3-small
APP_MIN_RELEVANCE=0.55
```

`OPENAI_BASE_URL` can target a compatible API. A separate embedding endpoint/key can be set with `OPENAI_EMBED_BASE_URL` and `OPENAI_EMBED_API_KEY`. With `MODEL_PROVIDER=openai`, the same compatible provider can supply chat through `OPENAI_CHAT_MODEL`.

Embedding model/provider changes use separate collections. Re-upload documents and prepare the research corpus for that configuration. Recalibrate retrieval thresholds when changing models.

## Provider availability

The public demo never calls OpenAI embeddings. If Ollama Cloud is unavailable, PDF retrieval and source-only answers continue; live attack evaluation requires generation. The recorded project experiment is labelled and kept separate from fresh results.

In optional semantic mode, HTTP 429 can indicate account quota or temporary throttling. Known quota/billing errors require account action. Identified transient rate limits receive bounded retries that respect `Retry-After`; partial corpus setup resumes without re-embedding completed batches. See the [OpenAI error-code reference](https://developers.openai.com/api/docs/guides/error-codes).
