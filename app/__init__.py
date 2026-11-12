"""PDF question-answering app built on top of the RAG-TEST research pipeline.

Layout (one responsibility per module):

    config.py        settings, read from environment variables
    embeddings.py    nomic-embed-text v1.5 with its task prefixes
    documents.py     PDF validation, text extraction, page-aware chunking
    vector_store.py  ChromaDB wrapper (add / list / delete / search)
    rag.py           the RAG pipeline: rewrite -> retrieve -> gate -> generate -> cite
    services.py      wires real Ollama models into the pipeline (used by the API)
    api.py           FastAPI HTTP endpoints
    schemas.py       request / response models for the API

The research code (main.py, rag_utils.py, suffix_attack/, ...) is untouched;
this package imports rag_utils.py instead of copying it.
"""
