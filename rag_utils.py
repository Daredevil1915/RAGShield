import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL       = "mistral"
DEFAULT_EMBED_MODEL     = "nomic-embed-text"
DEFAULT_TEMPERATURE     = 0.1

# ── CPU-only tuning ────────────────────────────────────────────────────────
# There is no GPU here, so the two things that actually cost time are
# (a) re-loading a ~4 GB model into RAM and (b) generating tokens nobody reads.
DEFAULT_NUM_THREAD  = 6      # physical cores (Ryzen 5 5600H is 6C/12T; SMT hurts llama.cpp)
DEFAULT_NUM_CTX     = 4096   # pinned, so prompt truncation can never surprise us mid-demo
DEFAULT_NUM_PREDICT = 256    # a RAG answer is 1-3 sentences; caps the runaway case
DEFAULT_KEEP_ALIVE  = 3600   # seconds; keep both models resident across all three acts


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def get_ollama_headers() -> dict:
    token = os.getenv("OLLAMA_API_KEY", "")
    return {"Authorization": f"Bearer {token}"} if token else {}

def get_llm_model() -> str:
    return os.getenv("OLLAMA_LLM_MODEL", DEFAULT_LLM_MODEL)

def get_embed_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)

def get_temperature() -> float:
    try:
        return float(os.getenv("OLLAMA_TEMPERATURE", DEFAULT_TEMPERATURE))
    except (ValueError, TypeError):
        return DEFAULT_TEMPERATURE

def _get_int(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, default))
    except (ValueError, TypeError):
        return default

def get_keep_alive() -> int:
    """Seconds to keep a model resident. OllamaEmbeddings only accepts an int."""
    return _get_int("OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)


# ── Terminal-safe display ──────────────────────────────────────────────────
# The MS-MARCO passages were ingested as UTF-8 bytes that had been decoded as
# latin-1, so the stored text is mojibake: 'Huītzilōpōchtli' is held as
# 'HuÄ«tzilÅ\x8dpÅ\x8dchtli'.  That leaves raw C1 bytes (U+0080-U+009F) in the
# text, and a terminal reads several of those as escape sequences — U+009D is
# OSC, which makes the terminal swallow *everything printed after it* until a
# string terminator arrives.  Printing such a passage looks exactly like the
# program hanging, even though it runs to completion.  Never print raw corpus
# text; route it through safe_display() first.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def repair_mojibake(text: str) -> str:
    """Undo the utf-8-read-as-latin-1 damage, when the text round-trips cleanly."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def safe_display(text: str, repair: bool = True) -> str:
    """Return `text` with control characters removed so it is safe to print."""
    if repair:
        text = repair_mojibake(text)
    return _CTRL_RE.sub("", text)


def ensure_ollama_server(base_url: str) -> None:
    health_url = f"{base_url}/api/tags"
    try:
        with urlopen(Request(health_url, headers=get_ollama_headers()), timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Ollama returned HTTP {response.status} at {health_url}."
                )
    except (HTTPError, URLError) as exc:
        raise RuntimeError(
            f"\n[ERROR] Ollama is not reachable at {base_url}.\n"
            f"  → Start Ollama (system tray or `ollama serve`) and try again.\n"
            f"  → Details: {exc}"
        ) from exc


def create_embeddings(base_url: str = None) -> OllamaEmbeddings:
    resolved_url = base_url or get_ollama_base_url()
    ensure_ollama_server(resolved_url)
    return OllamaEmbeddings(
        model=get_embed_model(),
        base_url=resolved_url,
        client_kwargs={"headers": get_ollama_headers(), "timeout": 300},
        num_thread=_get_int("OLLAMA_NUM_THREAD", DEFAULT_NUM_THREAD),
        keep_alive=get_keep_alive(),
    )


def create_llm(base_url: str = None) -> OllamaLLM:
    resolved_url = base_url or get_ollama_base_url()
    ensure_ollama_server(resolved_url)
    return OllamaLLM(
        model=get_llm_model(),
        base_url=resolved_url,
        client_kwargs={"headers": get_ollama_headers(), "timeout": 300},
        temperature=get_temperature(),
        num_thread=_get_int("OLLAMA_NUM_THREAD", DEFAULT_NUM_THREAD),
        num_ctx=_get_int("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX),
        num_predict=_get_int("OLLAMA_NUM_PREDICT", DEFAULT_NUM_PREDICT),
        keep_alive=get_keep_alive(),
    )


def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def create_mmr_retriever(vectorstore, k: int = 10, fetch_k: int = 30, lambda_mult: float = 0.6):
    # NOTE: Changed from "mmr" to "similarity" to bypass deduplication and enable flooding!
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )
