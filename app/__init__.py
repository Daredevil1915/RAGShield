"""PDF question-answering app built on top of the RAG-TEST research pipeline.

Layout (one responsibility per module):

    config.py        settings, read from environment variables
    embeddings.py    nomic-embed-text v1.5 with its task prefixes
    documents.py     PDF validation, text extraction, page-aware chunking
    vector_store.py  ChromaDB wrapper (add / list / delete / search)
    rag.py           the RAG pipeline: rewrite -> retrieve -> gate -> generate -> cite
    services.py      wires configured models into the pipeline (used by the API)
    model_backend.py hosted chat and embedding API adapters
    api.py           FastAPI HTTP endpoints
    schemas.py       request / response models for the API

The app and research share the model factories in rag_utils.py.
"""
