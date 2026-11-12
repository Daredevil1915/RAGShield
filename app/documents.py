"""PDF validation, text extraction and page-aware chunking.

Pipeline for one upload:

    raw bytes --validate--> pypdf reader --extract per page--> [PageText]
              --split each page--> [Document chunk with {doc_id, source, page}]

Chunks are cut *within* a page, never across pages, so every chunk has exactly
one page number. That is what makes "document X, page N" citations possible.
The trade-off is that a sentence running over a page break is split in two.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

from rag_utils import create_text_splitter, safe_display  # reused from the research code

MIN_TOTAL_TEXT_CHARS = 20  # less than this means a scanned or empty PDF


class InvalidDocumentError(ValueError):
    """The upload cannot be indexed. status_code is the HTTP code the API returns."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class PageText:
    page: int  # 1-based, the number a human sees in a PDF viewer
    text: str


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    num_pages: int
    pages: List[PageText] = field(default_factory=list)  # only pages that have text


def compute_doc_id(data: bytes) -> str:
    """Content hash, so uploading the same file twice is detected (idempotent upload)."""
    return hashlib.sha256(data).hexdigest()[:16]


def clean_text(text: str) -> str:
    text = safe_display(text, repair=False)          # strip control chars (rag_utils)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join words hyphenated at line ends
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(filename: str, data: bytes, max_bytes: int, max_pages: int) -> ParsedDocument:
    """Validate an uploaded file and extract its text page by page.

    Raises InvalidDocumentError with a user-facing message for every way an
    upload can be bad, so the API can return a clean 4xx instead of a 500.
    """
    if not filename or not filename.lower().endswith(".pdf"):
        raise InvalidDocumentError("Only PDF files are supported.", status_code=415)
    if not data:
        raise InvalidDocumentError("The file is empty.")
    if len(data) > max_bytes:
        raise InvalidDocumentError(
            f"File is too large ({len(data) / 1e6:.1f} MB). The limit is {max_bytes / 1e6:.0f} MB.",
            status_code=413,
        )
    # A real PDF starts with "%PDF-" (the spec allows a little junk before it).
    # This catches files that were renamed to .pdf.
    if b"%PDF-" not in data[:1024]:
        raise InvalidDocumentError("This file is not a valid PDF (missing PDF header).", status_code=415)

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Many "encrypted" PDFs only restrict printing and open with an empty password.
            try:
                if not reader.decrypt(""):
                    raise InvalidDocumentError("This PDF is password-protected. Remove the password and try again.")
            except InvalidDocumentError:
                raise
            except Exception:
                raise InvalidDocumentError("This PDF is password-protected. Remove the password and try again.")
        num_pages = len(reader.pages)
    except InvalidDocumentError:
        raise
    except Exception as exc:  # pypdf raises several exception types for broken files
        raise InvalidDocumentError(f"Could not read the PDF; it may be corrupted ({type(exc).__name__}).") from exc

    if num_pages == 0:
        raise InvalidDocumentError("The PDF has no pages.")
    if num_pages > max_pages:
        raise InvalidDocumentError(f"The PDF has {num_pages} pages; the limit is {max_pages}.", status_code=413)

    pages: List[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text = ""  # one unreadable page should not sink the whole document
        if text:
            pages.append(PageText(page=i, text=text))

    if sum(len(p.text) for p in pages) < MIN_TOTAL_TEXT_CHARS:
        raise InvalidDocumentError(
            "No extractable text found. The PDF is probably scanned images; OCR is not supported."
        )

    return ParsedDocument(doc_id=compute_doc_id(data), filename=filename, num_pages=num_pages, pages=pages)


def chunk_document(doc: ParsedDocument, chunk_size: int, chunk_overlap: int) -> tuple[list[Document], list[str]]:
    """Split each page into overlapping chunks. Returns (chunks, stable ids)."""
    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # rag_utils
    chunks: list[Document] = []
    ids: list[str] = []
    for page in doc.pages:
        for idx, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={"doc_id": doc.doc_id, "source": doc.filename, "page": page.page, "chunk": idx},
                )
            )
            ids.append(f"{doc.doc_id}-p{page.page}-c{idx}")
    return chunks, ids
