"""A reproducible, first-party PDF for the public interview workspace."""
from datetime import datetime, timezone
from functools import lru_cache

from fpdf import FPDF

DEMO_FILENAME = "RAGShield-Guide.pdf"
PAGES = [
    ("RAGShield: document question answering",
     "RAGShield lets visitors upload PDFs and ask questions with document and page citations. "
     "The pipeline parses each PDF page, splits text into overlapping chunks, stores vectors in ChromaDB, "
     "retrieves relevant passages, and asks Ollama Cloud to answer from that evidence. "
     "Source passages remain visible so readers can inspect the evidence behind an answer. "
     "The public demo uses normalized hashed word features for search on the Render server. "
     "This lightweight retrieval matches shared vocabulary; it is not a learned semantic embedding model. "
     "It needs no OpenAI credits, model download, or running developer laptop. "
     "Hosted semantic embeddings are an optional configuration for the full application."),
    ("Security research: retrieval versus jamming",
     "Retrieval success means an injected blocker document reaches the context shown to the language model. "
     "Jamming success means the clean pipeline answered the question but the poisoned pipeline refuses "
     "or no longer conveys the correct answer. A retrieved blocker does not necessarily jam an answer. "
     "RAGShield compares clean and poisoned responses using an isolated experimental collection, "
     "then removes injected blockers. Black-box optimization generates blockers using model calls; "
     "adversarial suffix experiments can use prebuilt blocker documents. "
     "The live demo corpus contains 24 gold passages and 40 distractors from the project's MS-MARCO subset. "
     "This 64-passage demonstration is different from the original 50,024-passage benchmark. "
     "The recorded example in the research workspace is a prior project execution, clearly separated from live runs."),
    ("Reliability, limitations, and deployment",
     "Ollama Cloud generates live answers without relying on the developer's computer. "
     "If the cloud generation service is unavailable, document Q&A can display exact sentences "
     "from retrieved passages with citations. This is labelled source-only mode, not a generated LLM answer. "
     "Research experiments require a live model and never substitute fabricated attack outcomes. "
     "Demo search relies on vocabulary overlap and can miss paraphrases or synonyms. "
     "Relevance thresholds and lexical grounding checks are heuristics, not guarantees of factual correctness "
     "or prompt-injection resistance. Free Render hosting may take time to wake after inactivity, "
     "and uploaded indexes can be lost when its temporary filesystem resets. "
     "The built-in guide and sample research corpus can be recreated without external embedding API calls."),
]


@lru_cache(maxsize=1)
def demo_pdf() -> bytes:
    pdf = FPDF()
    pdf.set_creation_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    pdf.set_title("RAGShield field guide")
    for title, text in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.multi_cell(0, 9, title)
        pdf.ln(5)
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 7, text)
    return bytes(pdf.output())


def load_demo_document(services) -> dict:
    from .services import index_pdf
    return index_pdf(services.store, services.settings, DEMO_FILENAME, demo_pdf())
