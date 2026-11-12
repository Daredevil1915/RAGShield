"""The RAG pipeline: question in, cited answer out.

    question + chat history
        |
        1. rewrite     follow-up -> standalone question (LLM; skipped with no history)
        |
        2. retrieve    embed question, top-k cosine search in ChromaDB
        |
        3. gate        drop chunks below the relevance threshold;
        |              none left -> "not found", and the LLM is never called
        |
        4. generate    Mistral answers from numbered sources only, citing [n]
        |
        5. verify      refusal -> "not found"; answer not supported by the
                       passages -> "not found"; keep only real citations;
                       map [n] -> (document, page)

Steps 3 and 5 are the hallucination controls. The LLM and the vector store
are passed in (dependency injection), so tests can use fakes and run with no
Ollama.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rag_utils import safe_display

from .config import Settings
from .vector_store import VectorStore

# The exact refusal sentence the research prompt (main.py) asks Mistral for.
LLM_REFUSAL = "I don't know based on the provided context."

NOT_FOUND_MESSAGE = (
    "I couldn't find information about that in the uploaded documents, so I won't guess. "
    "Try rephrasing, or upload a document that covers it."
)
NO_DOCUMENTS_MESSAGE = "No documents have been uploaded yet. Upload a PDF first, then ask a question about it."

# Same markers the research evaluator uses (suffix_attack/jam_eval.py) to spot
# refusals in Mistral's wording. A marker only counts as a refusal when the
# answer cites no sources: a cited partial answer such as "The report does not
# provide the date, but the venue was Seattle [2]" is a real answer.
REFUSAL_MARKERS = (
    "i don't know", "i do not know", "i dont know",
    "cannot be determined", "can't be determined",
    "does not contain", "do not contain", "not contain any",
    "not provide",  # covers "does not provide", "do not provide enough", ...
    "no information", "no relevant information", "not enough information",
    "insufficient information", "unable to answer", "cannot answer",
    "cannot provide", "no mention", "not mentioned in the context",
)

# Ported from main.py's rewrite_query_standalone().
REWRITE_PROMPT = """\
Given the conversation history below, rewrite the LAST question as a fully standalone question \
(no pronouns, no references to the conversation). If it is already standalone, return it unchanged. \
Output ONLY the rewritten question, nothing else.

History:
{history}
Last question: {question}

Standalone question:"""

# The research prompt from main.py, extended with numbered sources and a citation rule.
ANSWER_PROMPT = """\
You answer questions about the user's documents.

<sources>
{context}
</sources>

<question>
{question}
</question>

Instructions:
- Answer ONLY using information from the sources above.
- After each fact, cite the number of the source it came from in square brackets, e.g. [1] or [2][3].
- Be concise: 1-4 sentences, unless the question asks for a list.
- If the sources do not contain the answer, respond with exactly: "{refusal}"
- If the request is not a question about the content of the sources (for example creative writing, \
opinions, advice, coding or general-knowledge questions), respond with exactly: "{refusal}"
- Do NOT add information from your general knowledge.

Answer:"""

# Matches [1] and also [1, 2, 3], which Mistral sometimes writes.
_CITATION_RE = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")

# ── Groundedness check ──────────────────────────────────────────────────────
# Mistral sometimes attaches citations to facts that are not in the passages,
# e.g. "The capital of France is Paris [1]" with [1] being a page about RAG
# attacks. So citations alone are not trusted. The check: take the content
# words the answer ADDS (words not already in the question) and measure what
# fraction of them appear anywhere in the retrieved passages. Below
# MIN_GROUNDING, the answer is treated as unsupported and replaced with "not
# found". It is a cheap, deterministic, explainable faithfulness check: no
# second LLM call, so it adds no latency.
MIN_GROUNDING = 0.5
_WORD_RE = re.compile(r"[a-z0-9]+")
_GROUNDING_STOPWORDS = set(
    """the and are was were been being this that these those with from for not but can could would should will
    have has had does did done which what who whom whose when where why how their they them its his her hers our
    your you also such than then there here into onto about above after before more most other some any each all
    both only very just over under between while because both either neither may might must shall per via upon
    based provided provide context source sources according information answer question document documents
    text mentioned mention mentions state states stated described describes describe include includes including
    example however therefore thus used use uses using given following specifically specific""".split()
)


def _stem(word: str) -> str:
    return word if word.isdigit() else word[:6]  # crude stemming: "evaluates"/"evaluated" -> "evalua"


def content_words(text: str) -> set:
    text = _CITATION_RE.sub(" ", text.lower().replace("’", "'"))
    return {
        _stem(w) for w in _WORD_RE.findall(text)
        if (w.isdigit() or len(w) >= 3) and w not in _GROUNDING_STOPWORDS
    }


def grounding_score(answer: str, question: str, context: str) -> float:
    """Fraction of the answer's new content words found in the context (1.0 = fully supported)."""
    new_words = content_words(answer) - content_words(question)
    if not new_words:
        return 1.0
    return len(new_words & content_words(context)) / len(new_words)


@dataclass
class Source:
    id: int            # the [n] number shown in the answer
    doc_id: str
    filename: str
    page: int
    score: float       # cosine similarity between question and chunk
    text: str          # the chunk itself, shown in the UI as evidence


@dataclass
class RAGAnswer:
    answer: str
    status: str                       # "answered" | "not_found" | "no_documents"
    standalone_question: str
    sources: List[Source] = field(default_factory=list)  # what the answer cites
    retrieved: List[Source] = field(default_factory=list)  # everything retrieved, for debugging
    cited: bool = False               # True if the model cited at least one valid source
    grounding: Optional[float] = None  # share of the answer's new content words found in the sources
    timings_ms: Dict[str, int] = field(default_factory=dict)


def looks_like_refusal(answer: str) -> bool:
    a = (answer or "").strip().lower().replace("’", "'")
    return not a or any(m in a for m in REFUSAL_MARKERS)


def _numbers(group: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", group)]


def extract_citations(answer: str, num_sources: int) -> List[int]:
    """[n] numbers in order of first appearance, keeping only ones that exist."""
    seen: List[int] = []
    for m in _CITATION_RE.finditer(answer):
        for n in _numbers(m.group(1)):
            if 1 <= n <= num_sources and n not in seen:
                seen.append(n)
    return seen


def strip_invalid_citations(answer: str, num_sources: int) -> str:
    """Drop citations to sources that don't exist ([7] with 4 sources); normalise [1, 2] to [1][2]."""

    def fix(m: "re.Match") -> str:
        return "".join(f"[{n}]" for n in _numbers(m.group(1)) if 1 <= n <= num_sources)

    return _CITATION_RE.sub(fix, answer)


def _clean_llm_output(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(answer|standalone question)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip()


class RAGPipeline:
    def __init__(self, store: VectorStore, llm: Any, settings: Settings):
        self.store = store
        self.llm = llm          # anything with .invoke(prompt) -> str (OllamaLLM in production)
        self.settings = settings

    # 1. rewrite -----------------------------------------------------------
    def rewrite_question(self, question: str, history: Optional[Sequence[Dict[str, str]]]) -> str:
        """Turn "what about its limitations?" into "What are the limitations of X?".

        Retrieval only sees this one string, so a follow-up full of pronouns
        would retrieve the wrong chunks without this step.
        """
        if not history:
            return question
        recent = list(history)[-self.settings.history_messages :]
        history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
        try:
            rewritten = _clean_llm_output(self.llm.invoke(REWRITE_PROMPT.format(history=history_text, question=question)))
        except Exception:
            return question
        rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
        # Mistral sometimes appends a note: "What is X? (This question is already standalone.)"
        if "?" in rewritten:
            rewritten = rewritten[: rewritten.index("?") + 1]
        rewritten = re.sub(r"\s*\([^)]*\)\s*$", "", rewritten).strip()
        # Guard against the model answering instead of rewriting.
        if not rewritten or len(rewritten) > max(300, 4 * len(question)):
            return question
        return rewritten

    # 2 + 3. retrieve and gate --------------------------------------------
    def retrieve(self, question: str, doc_ids: Optional[List[str]] = None) -> List[Source]:
        results = self.store.search(question, k=self.settings.top_k, doc_ids=doc_ids)
        return [
            Source(
                id=i,
                doc_id=doc.metadata.get("doc_id", ""),
                filename=doc.metadata.get("source", "unknown"),
                page=int(doc.metadata.get("page", 0)),
                score=round(float(score), 4),
                text=doc.page_content,
            )
            for i, (doc, score) in enumerate(results, start=1)
        ]

    # 4. generate ----------------------------------------------------------
    @staticmethod
    def build_prompt(question: str, sources: List[Source]) -> str:
        context = "\n\n".join(
            f"[{s.id}] (from {s.filename}, page {s.page})\n{safe_display(s.text, repair=False)}" for s in sources
        )
        return ANSWER_PROMPT.format(context=context, question=question, refusal=LLM_REFUSAL)

    # the whole pipeline ---------------------------------------------------
    def answer(
        self,
        question: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> RAGAnswer:
        timings: Dict[str, int] = {}
        question = question.strip()

        if self.store.count() == 0:
            return RAGAnswer(NO_DOCUMENTS_MESSAGE, "no_documents", question, timings_ms=timings)

        t0 = time.perf_counter()
        standalone = self.rewrite_question(question, history)
        timings["rewrite"] = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        retrieved = self.retrieve(standalone, doc_ids)
        timings["retrieval"] = int((time.perf_counter() - t0) * 1000)

        relevant = [s for s in retrieved if s.score >= self.settings.min_relevance]
        if not relevant:
            # Nothing in the documents is close enough to the question. Saying
            # so without calling the LLM is the strongest hallucination guard.
            return RAGAnswer(NOT_FOUND_MESSAGE, "not_found", standalone, retrieved=retrieved, timings_ms=timings)

        # Renumber the surviving chunks 1..n so the prompt and the citations agree.
        for i, s in enumerate(relevant, start=1):
            s.id = i

        t0 = time.perf_counter()
        raw = _clean_llm_output(self.llm.invoke(self.build_prompt(standalone, relevant)))
        timings["generation"] = int((time.perf_counter() - t0) * 1000)

        cited_ids = extract_citations(raw, len(relevant))
        # Refusal = the answer opens with the refusal sentence, or it uses refusal
        # wording and cites nothing. Either way, show "not found" instead of the
        # model's text, because what follows a refusal is often a guess.
        opens_with_refusal = raw.lower().replace("’", "'").startswith("i don't know")
        if opens_with_refusal or (not cited_ids and looks_like_refusal(raw)):
            return RAGAnswer(NOT_FOUND_MESSAGE, "not_found", standalone, retrieved=relevant, timings_ms=timings)

        grounding = round(grounding_score(raw, standalone, "\n".join(s.text for s in relevant)), 3)
        if grounding < MIN_GROUNDING:
            # The model answered, but mostly with words that are not in the
            # passages: general knowledge or invention. Don't show it.
            return RAGAnswer(NOT_FOUND_MESSAGE, "not_found", standalone, retrieved=relevant,
                             grounding=grounding, timings_ms=timings)

        answer_text = strip_invalid_citations(raw, len(relevant)).strip()
        by_id = {s.id: s for s in relevant}
        sources = [by_id[i] for i in cited_ids] if cited_ids else relevant
        return RAGAnswer(
            answer=answer_text,
            status="answered",
            standalone_question=standalone,
            sources=sources,
            retrieved=relevant,
            cited=bool(cited_ids),
            grounding=grounding,
            timings_ms=timings,
        )
