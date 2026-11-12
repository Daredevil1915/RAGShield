"""Request and response models. FastAPI validates every request against these
(422 on bad input) and generates the interactive docs at /docs from them."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import get_settings

MAX_QUESTION_CHARS = get_settings().max_question_chars


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    # The client sends the conversation so far; the server keeps no session state.
    history: List[ChatMessage] = Field(default_factory=list, max_length=50)
    # Optional: only search these documents. None or empty = all documents.
    doc_ids: Optional[List[str]] = None

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty.")
        return v


class SourceOut(BaseModel):
    id: int
    doc_id: str
    filename: str
    page: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    status: Literal["answered", "not_found", "no_documents"]
    standalone_question: str
    sources: List[SourceOut]
    cited: bool
    grounding: Optional[float] = None
    timings_ms: Dict[str, int]
    generation_mode: Literal["llm", "extractive"] = "llm"


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    pages_with_text: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_pages: int
    pages_with_text: int
    chunks: int
    duplicate: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "configured", "degraded"]
    ollama: Optional[Dict] = None
    backend: Optional[Dict] = None
    documents: Optional[int] = None
