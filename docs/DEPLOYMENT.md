# Deployment and configuration

[Back to RAGShield](../README.md)

## Public interview demo

The default hosted configuration prioritizes a usable visitor experience:

- A built-in three-page PDF is indexed automatically.
- Word-feature retrieval runs on Render with no embedding API, credits or model downloads.
- Document Q&A returns exact retrieved sentences with citations and a visible **Source-only answer** label.
- The 64-passage research corpus is prepared automatically. Live experiments inject a blocker and compare real retrieval contexts, then restore the collection. They do not claim to evaluate LLM jamming.
- A recorded LLM project run is labelled separately. Optional cloud generation and full LLM experiments can be enabled by the owner.

Word-feature retrieval uses vocabulary overlap. It is intentionally lighter than a semantic embedding model and may miss paraphrases. Research never replaces a failed live run with a fabricated outcome.

## Deploy on Render

[Deploy to Render](https://render.com/deploy?repo=https://github.com/Daredevil1915/RAGShield)

1. Create a Blueprint from this repository. Render reads `render.yaml`.
2. No model API credentials are required for the default Blueprint. Optional Ollama Cloud credentials can be added later.
3. Open the service URL. There is no login or visitor configuration step.
4. Try a suggested question about the guide, upload a PDF, or open Security Research and select **Run retrieval experiment**. Edit the blocker text or context depth to explore how retrieval changes.

| Variable | Public demo value | Purpose |
| --- | --- | --- |
| `APP_PUBLIC_DEMO` | `1` | Enables no-key retrieval, the sample guide and source-only fallback |
| `APP_DEMO_LIVE_LLM` | `0` | Keeps the visitor workflow independent of model API billing |
| `MODEL_PROVIDER` | `ollama_cloud` | Provider used if optional live generation is enabled |
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

Open http://localhost:10000. Your own computer is not needed once the app is deployed on Render. Store any optional provider secrets in environment settings; never commit them.

## Optional live LLM demonstration

Set `APP_DEMO_LIVE_LLM=1` and configure `OLLAMA_API_KEY` from [Ollama account keys](https://ollama.com/settings/keys) with working provider access. This enables live cloud generation and full clean-versus-poisoned LLM experiments while retaining no-key word-feature retrieval. Provider billing and rate limits apply in this optional mode. Document Q&A falls back to labelled source excerpts if generation fails; research never fabricates a model verdict.

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

The default public demo makes no model API calls. Source-excerpt Q&A and retrieval-poisoning experiments run on Render. The optional live LLM mode depends on its configured provider; recorded project results are labelled and kept separate from fresh results.

In optional semantic mode, HTTP 429 can indicate account quota or temporary throttling. Known quota/billing errors require account action. Identified transient rate limits receive bounded retries that respect `Retry-After`; partial corpus setup resumes without re-embedding completed batches. See the [OpenAI error-code reference](https://developers.openai.com/api/docs/guides/error-codes).
