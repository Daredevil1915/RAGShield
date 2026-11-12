"""Hosted model adapters shared by document Q&A and research workflows."""
from __future__ import annotations

import hashlib
import math
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx
from langchain_core.embeddings import Embeddings


class ModelBackendError(RuntimeError):
    """A safe model error that can be displayed without provider credentials."""


def get_model_provider() -> str:
    provider = os.getenv("MODEL_PROVIDER", "ollama_cloud").strip().lower()
    if provider not in {"ollama", "ollama_cloud", "openai"}:
        raise ModelBackendError("MODEL_PROVIDER must be ollama_cloud, openai or ollama.")
    return provider


def get_embedding_provider() -> str:
    default = "ollama" if get_model_provider() == "ollama" else "openai"
    provider = os.getenv("EMBEDDING_PROVIDER", default).strip().lower()
    if provider not in {"ollama", "openai"}:
        raise ModelBackendError("EMBEDDING_PROVIDER must be openai or ollama.")
    if get_model_provider() != "ollama" and provider == "ollama":
        raise ModelBackendError("Cloud mode requires EMBEDDING_PROVIDER=openai; local Ollama is not used.")
    return provider


def cloud_config(kind: str) -> tuple[str, str, str]:
    """Return URL, key and model. Embeddings can use a separate hosted provider."""
    if kind == "chat" and get_model_provider() == "ollama_cloud":
        return (os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com").rstrip("/"),
                os.getenv("OLLAMA_API_KEY", "").strip(),
                os.getenv("OLLAMA_CLOUD_MODEL", "gemma4:31b"))
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if kind == "embeddings":
        return (os.getenv("OPENAI_EMBED_BASE_URL", base).rstrip("/"),
                os.getenv("OPENAI_EMBED_API_KEY", key).strip(),
                os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    return base, key, os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")


def require_cloud_config(kind: str) -> tuple[str, str, str]:
    base, key, model = cloud_config(kind)
    if not key:
        setting = "OLLAMA_API_KEY" if kind == "chat" and get_model_provider() == "ollama_cloud" else "OPENAI_API_KEY (or OPENAI_EMBED_API_KEY for embeddings)"
        raise ModelBackendError(f"Set {setting} in the service environment.")
    parts = urlsplit(base)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ModelBackendError("Model API base URL must be an HTTP(S) URL without credentials, query or fragment.")
    if parts.scheme != "https" and parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ModelBackendError("Use HTTPS for hosted model APIs.")
    if not model.strip():
        raise ModelBackendError(f"Configure a model for {kind}.")
    return base, key, model


def embedding_namespace() -> str:
    """Do not reuse vectors across embedding models, endpoints or prefix schemes."""
    if get_embedding_provider() == "ollama":
        identity = ["ollama", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                    os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")]
    else:
        base, _, model = cloud_config("embeddings")
        identity = ["openai", base, model]
    return hashlib.sha256("\n".join(identity).encode()).hexdigest()[:12]


def _post(kind: str, path: str, payload: dict) -> dict:
    base, key, _ = require_cloud_config(kind)
    for attempt in range(3):
        try:
            response = httpx.post(base + path, headers={"Authorization": f"Bearer {key}"},
                                  json=payload, timeout=httpx.Timeout(120, connect=10))
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError
            return data
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            try:
                error = exc.response.json().get("error", {})
                error = error if isinstance(error, dict) else {}
            except (ValueError, AttributeError):
                error = {}
            # Only map known codes. Never display arbitrary provider messages.
            quota_hints = {
                "insufficient_quota": "API quota is unavailable or exhausted. Check the provider's API billing and project limits. Retrying alone will not restore access.",
                "credit_balance_exhausted": "API credits are exhausted. Add credits in the provider's API billing settings before retrying.",
                "billing_not_active": "API billing is not active. Enable it in the provider's account before retrying.",
                "project_spend_limit_exceeded": "The API project's spending limit has been reached. Check that project's billing limits before retrying.",
                "organization_spend_limit_exceeded": "The organization's API spending limit has been reached. Check its billing limits before retrying.",
                "organization_usage_limit_exceeded": "The organization's API usage limit has been reached. Request a higher provider limit or wait for the reset.",
                "usage_limit_exceeded": "An API usage or billing limit has been reached. Check the provider account's limits before retrying.",
            }
            reason = error.get("code") if isinstance(error.get("code"), str) else None
            error_type = error.get("type") if isinstance(error.get("type"), str) else None
            if reason not in quota_hints and error_type in quota_hints:
                reason = error_type
            if reason in quota_hints:
                raise ModelBackendError(f"Cloud {kind} API returned HTTP {code}. {quota_hints[reason]}") from None
            transient = code == 429 and reason in {"rate_limit_exceeded", "slow_down"}
            delay = _retry_delay(exc.response, attempt)
            if transient and attempt < 2 and delay <= 10:
                time.sleep(delay)
                continue
            hint = {401: "Check the API key.", 403: "Check model access.",
                    404: "Check the API base URL and model name.",
                    429: "The provider did not identify whether this is a rate or quota limit. Check API billing and limits before retrying."}.get(code, "Try again later.")
            if transient:
                hint = (f"Temporary rate limit. Wait at least {math.ceil(delay)} seconds before trying again."
                        if delay > 10 else "Temporary rate limit remains after automatic retries. Try again shortly.")
            raise ModelBackendError(f"Cloud {kind} API returned HTTP {code}. {hint}") from None
        except (httpx.HTTPError, ValueError):
            raise ModelBackendError(f"Cloud {kind} API could not be reached or returned invalid JSON.") from None


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    value = response.headers.get("Retry-After")
    if value:
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                return max(0, seconds)
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                return max(0, (when - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return 0.5 * (2 ** attempt) + random.uniform(0, 0.25)


class CloudLLM:
    """The research code expects invoke(prompt) to return plain text."""

    def __init__(self):
        self.model = require_cloud_config("chat")[2]

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if get_model_provider() == "ollama_cloud":
            result = _post("chat", "/api/chat", {"model": self.model, "messages": messages,
                           "stream": False, "think": False,
                           "options": {"temperature": 0, "num_predict": 512}})
            message = result.get("message")
            content = message.get("content") if isinstance(message, dict) else None
        else:
            result = _post("chat", "/chat/completions", {"model": self.model, "messages": messages,
                           "stream": False, "temperature": 0, "max_tokens": 512})
            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                content = None
        if not isinstance(content, str) or not content.strip():
            raise ModelBackendError("Cloud chat API returned no answer text. Check the selected model and token limit.")
        return content


class CloudEmbeddings(Embeddings):
    def __init__(self):
        self.model = require_cloud_config("embeddings")[2]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = _post("embeddings", "/embeddings", {"model": self.model, "input": texts, "encoding_format": "float"})
        try:
            rows = sorted(result["data"], key=lambda row: row["index"])
            if [row["index"] for row in rows] != list(range(len(texts))):
                raise ValueError
            vectors = [row["embedding"] for row in rows]
            if not vectors or not vectors[0] or any(len(v) != len(vectors[0]) for v in vectors):
                raise ValueError
            if any(not isinstance(n, (int, float)) or not math.isfinite(n) for v in vectors for n in v):
                raise ValueError
            return vectors
        except (KeyError, TypeError, ValueError):
            raise ModelBackendError("Cloud embeddings API returned invalid vectors.") from None

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def cloud_status() -> dict:
    """Configuration check only: health polling must not incur inference charges."""
    result = {"provider": get_model_provider(), "configured": False,
              "models": {"llm": False, "embeddings": False}}
    try:
        get_embedding_provider()
        require_cloud_config("chat")
        require_cloud_config("embeddings")
        result.update(configured=True, models={"llm": True, "embeddings": True},
                      llm_model=cloud_config("chat")[2], embedding_model=cloud_config("embeddings")[2])
    except ModelBackendError as exc:
        result["error"] = str(exc)
    return result
