"""The RAG pipeline: retrieval, citations, follow-ups and the hallucination guards."""
from app.rag import NO_DOCUMENTS_MESSAGE, NOT_FOUND_MESSAGE, extract_citations, looks_like_refusal
from app.services import index_pdf
from tests.conftest import FakeLLM


def test_answer_cites_document_and_page(pipeline, store, settings, physics_pdf, llm):
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = pipeline.answer("At what temperature does water boil at sea level?")
    assert res.status == "answered"
    assert res.cited
    assert res.sources[0].filename == "physics.pdf"
    assert res.sources[0].page == 1
    # the prompt numbers the sources and tells the model where each came from
    prompt = llm.answer_prompts[0]
    assert "[1] (from physics.pdf, page 1)" in prompt
    assert "Answer ONLY using information from the sources" in prompt


def test_irrelevant_question_is_refused_without_calling_the_llm(pipeline, store, settings, physics_pdf, llm):
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = pipeline.answer("Who won the football world cup final?")
    assert res.status == "not_found"
    assert res.answer == NOT_FOUND_MESSAGE
    assert res.sources == []
    assert llm.prompts == []  # gated before generation: nothing to hallucinate from


def test_llm_refusal_becomes_not_found(store, settings, physics_pdf):
    llm = FakeLLM(lambda p: "I don't know based on the provided context.")
    from app.rag import RAGPipeline

    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = RAGPipeline(store, llm, settings).answer("What is the boiling point of water on Mars?")
    assert res.status == "not_found"
    assert res.sources == []


def test_refusal_followed_by_a_guess_is_still_not_found(store, settings, physics_pdf):
    from app.rag import RAGPipeline

    llm = FakeLLM(lambda p: "I don't know based on the provided context. However, it is probably 70 degrees [1].")
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = RAGPipeline(store, llm, settings).answer("What is the boiling point of water on Everest?")
    assert res.status == "not_found"


def test_cited_partial_answer_is_kept(store, settings, physics_pdf):
    from app.rag import RAGPipeline

    llm = FakeLLM(lambda p: "The text does not provide an exact altitude, but water boils at a lower temperature at high altitude [1].")
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = RAGPipeline(store, llm, settings).answer("Does water boil differently at high altitude?")
    assert res.status == "answered"
    assert res.cited


def test_made_up_citation_numbers_are_removed(store, settings, physics_pdf):
    from app.rag import RAGPipeline

    llm = FakeLLM(lambda p: "Water boils at 100 degrees Celsius [1][9].")
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = RAGPipeline(store, llm, settings).answer("At what temperature does water boil?")
    assert "[9]" not in res.answer
    assert [s.id for s in res.sources] == [1]


def test_uncited_answer_falls_back_to_retrieved_context(store, settings, physics_pdf):
    from app.rag import RAGPipeline

    llm = FakeLLM(lambda p: "Water boils at 100 degrees Celsius.")
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = RAGPipeline(store, llm, settings).answer("At what temperature does water boil?")
    assert res.status == "answered"
    assert not res.cited
    assert res.sources  # the UI shows these as "retrieved context"


def test_follow_up_question_is_rewritten_before_retrieval(store, settings, physics_pdf):
    from app.rag import RAGPipeline

    def reply(prompt):
        if "Standalone question:" in prompt:
            return "What does photosynthesis produce from sunlight and water?"
        return "Photosynthesis produces glucose and oxygen [1]."

    llm = FakeLLM(reply)
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    history = [
        {"role": "user", "content": "What is photosynthesis?"},
        {"role": "assistant", "content": "It converts sunlight into glucose [1]."},
    ]
    res = RAGPipeline(store, llm, settings).answer("And what does it produce?", history=history)
    assert len(llm.rewrite_prompts) == 1
    assert "What is photosynthesis?" in llm.rewrite_prompts[0]
    assert res.standalone_question == "What does photosynthesis produce from sunlight and water?"
    assert res.status == "answered"
    assert res.sources[0].page == 2


def test_no_rewrite_without_history(pipeline, store, settings, physics_pdf, llm):
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    pipeline.answer("At what temperature does water boil?")
    assert llm.rewrite_prompts == []


def test_bad_rewrite_falls_back_to_original_question(store, settings, physics_pdf):
    from app.rag import RAGPipeline

    long_ramble = "Sure! " + "Here is a long explanation about many things. " * 30
    llm = FakeLLM(lambda p: long_ramble if "Standalone question:" in p else "Water boils at 100 degrees [1].")
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = RAGPipeline(store, llm, settings).answer("water boil temperature?", history=[{"role": "user", "content": "hi"}])
    assert res.standalone_question == "water boil temperature?"


def test_rewrite_drops_notes_the_model_appends(store, settings, physics_pdf):
    """Real Mistral output seen in the UI check."""
    from app.rag import RAGPipeline

    note = "What is photosynthesis? (This question does not refer to the previous conversation.)"
    llm = FakeLLM(lambda p: note if "Standalone question:" in p else "It makes glucose and oxygen [1].")
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = RAGPipeline(store, llm, settings).answer("What is photosynthesis?", history=[{"role": "user", "content": "hi"}])
    assert res.standalone_question == "What is photosynthesis?"


def test_empty_knowledge_base(pipeline, llm):
    res = pipeline.answer("Anything?")
    assert res.status == "no_documents"
    assert res.answer == NO_DOCUMENTS_MESSAGE
    assert llm.prompts == []


def test_document_filter_restricts_retrieval(store, settings, physics_pdf, history_pdf):
    from app.rag import RAGPipeline

    pipeline = RAGPipeline(store, FakeLLM(lambda p: "It took place in 1066 between Normans and English forces [1]."), settings)
    physics = index_pdf(store, settings, "physics.pdf", physics_pdf)
    history = index_pdf(store, settings, "history.pdf", history_pdf)
    res = pipeline.answer("When did the Battle of Hastings take place?", doc_ids=[history["doc_id"]])
    assert res.status == "answered"
    assert {s.filename for s in res.retrieved} == {"history.pdf"}
    res = pipeline.answer("When did the Battle of Hastings take place?", doc_ids=[physics["doc_id"]])
    assert res.status == "not_found"


def test_fabricated_citation_is_caught_by_grounding_check(store, settings, physics_pdf):
    """Real failure seen with Mistral: a general-knowledge answer with a made-up citation."""
    from app.rag import RAGPipeline

    llm = FakeLLM(lambda p: "The capital of France is Paris [1].")
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    # "water" makes the question pass the relevance gate, so the LLM is called
    res = RAGPipeline(store, llm, settings).answer("What is the capital of France, a country with water?")
    assert llm.answer_prompts, "the question should pass the relevance gate"
    assert res.status == "not_found"
    assert res.grounding is not None and res.grounding < 0.5


def test_off_task_request_is_caught_by_grounding_check(store, settings, physics_pdf):
    from app.rag import RAGPipeline

    poem = "Oh water, shimmering sea so bright [1],\nwaves that dance in silver night [1]."
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    llm = FakeLLM(lambda p: poem)
    res = RAGPipeline(store, llm, settings).answer("Write a poem about water.")
    assert llm.answer_prompts, "the request should pass the relevance gate"
    assert res.status == "not_found"


def test_grounded_answer_reports_its_grounding(pipeline, store, settings, physics_pdf):
    index_pdf(store, settings, "physics.pdf", physics_pdf)
    res = pipeline.answer("At what temperature does water boil?")
    assert res.status == "answered"
    assert res.grounding >= 0.9


def test_duplicate_upload_is_not_indexed_twice(store, settings, physics_pdf):
    first = index_pdf(store, settings, "physics.pdf", physics_pdf)
    count = store.count()
    second = index_pdf(store, settings, "physics-copy.pdf", physics_pdf)
    assert not first["duplicate"] and second["duplicate"]
    assert store.count() == count


def test_citation_and_refusal_helpers():
    from app.rag import grounding_score, strip_invalid_citations

    assert extract_citations("A [2] b [1][2] c [7]", num_sources=3) == [2, 1]
    assert extract_citations("Several sources agree [1, 3].", num_sources=3) == [1, 3]
    assert strip_invalid_citations("x [1, 9] y [8]", num_sources=3) == "x [1] y "
    assert grounding_score("Paris [1]", "capital of France?", "RAG systems retrieve documents.") == 0.0
    assert grounding_score("It was 1066 [1].", "When was Hastings?", "The battle was in 1066.") == 1.0
    assert looks_like_refusal("I don’t know based on the provided context.")  # curly apostrophe
    assert looks_like_refusal("")
    assert not looks_like_refusal("Water boils at 100 degrees [1].")
