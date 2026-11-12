"""RAG Security Research Mode — orchestration tests.

These run without Ollama or the 50k vector copy. They verify the parts that
encode the report's methodology and the safety guarantees:
  * the valid-target dataset loads,
  * pre-made blockers are found from the real result files,
  * retrieval success is never conflated with jamming success,
  * injected blockers are ALWAYS cleaned up — even when the evaluator raises,
  * results export to JSON + CSV.
The real JamEvaluator (which needs Ollama) is replaced by a fake.
"""
import json

import pytest

from app.research import BLOCKER_TAG, ExperimentResult, ResearchService


# ── fakes ──────────────────────────────────────────────────────────────────
class FakeCol:
    """Minimal chromadb-collection stand-in tracking the live id-set."""

    def __init__(self, ids):
        self.ids = set(ids)
        self.deleted = []

    def get(self, include=None, where=None, limit=None, offset=None):
        if where:  # the blocker-tag filter used by sweep_cleanup
            return {"ids": [i for i in self.ids if i.startswith("blk")]}
        return {"ids": list(self.ids)}

    def delete(self, ids):
        for i in ids:
            self.ids.discard(i)
        self.deleted.extend(ids)


class FakeEval:
    def __init__(self, result=None, raise_exc=None, leak_col=None):
        self.use_llm_judge = True
        self.result = result
        self.raise_exc = raise_exc
        self.leak_col = leak_col

    def evaluate(self, query, blocker):
        if self.leak_col is not None:  # simulate injected blocker + an accidental stray
            self.leak_col.ids.update({"blk1", "stray1"})
        if self.raise_exc:
            raise self.raise_exc
        return dict(self.result, query=query, full_blocker=blocker)


def sample_eval(jammed=True, retrieved=True, outcome="refusal", valid=True):
    return {
        "clean_answer": "Huitzilopochtli is an Aztec deity of war and the sun.",
        "poisoned_answer": "I don't know based on the provided context." if jammed else "An Aztec deity of war.",
        "clean_refuses": not valid,
        "poisoned_refuses": jammed,
        "blocker_sim_rank": 1 if retrieved else -1,
        "blocker_ctx_pos": 2 if retrieved else -1,
        "retrieved": retrieved,
        "num_context": 5,
        "outcome": outcome,
        "valid_target": valid,
        "jammed": jammed,
    }


@pytest.fixture
def svc(tmp_path):
    s = ResearchService(export_dir=str(tmp_path / "runs"))
    return s


# ── dataset + pre-made blockers (real files) ────────────────────────────────
def test_dataset_has_valid_targets(svc):
    qs = svc.dataset()
    assert len(qs) >= 15
    assert any("huitzilopochtli" in q.query.lower() for q in qs)


def test_prebuilt_blocker_found_from_result_files(svc):
    # The report's control condition is saved for all 18 valid targets.
    styles = svc.available_prebuilt_styles("what is qd?")
    assert styles, "expected at least one pre-made blocker style for a valid target"
    blk = svc.get_prebuilt_blocker("what is qd?", styles[0])
    assert blk and blk["full_blocker"].strip()
    assert "pre-made" in blk["origin"]


def test_prebuilt_blocker_missing_returns_none(svc):
    assert svc.get_prebuilt_blocker("a query with no saved blocker at all", "distraction") is None


# ── methodology: retrieval is reported separately from jamming ───────────────
def test_retrieved_but_not_jammed_is_not_counted_as_jam():
    r = ExperimentResult(
        query="q", method="GCG", style="passive", blocker_origin="t", full_blocker="b",
        clean_answer="A", poisoned_answer="A", clean_refuses=False, poisoned_refuses=False,
        blocker_sim_rank=1, blocker_ctx_pos=1, retrieved=True, num_context=5,
        outcome="correct", valid_target=True, jammed=False,
        k=5, used_llm_judge=True, blockers_cleaned=0, ids_restored=0, elapsed_s=1.0, timestamp="t",
    )
    assert "NOT jammed" in r.verdict_label and "retrieved" in r.verdict_label
    assert "Retrieval: retrieved" in r.retrieval_vs_jamming
    assert "Jamming: not jammed" in r.retrieval_vs_jamming


def test_invalid_target_label():
    r = ExperimentResult(
        query="q", method="GCG", style="passive", blocker_origin="t", full_blocker="b",
        clean_answer="I don't know", poisoned_answer="I don't know", clean_refuses=True, poisoned_refuses=True,
        blocker_sim_rank=-1, blocker_ctx_pos=-1, retrieved=False, num_context=5,
        outcome="clean_refused", valid_target=False, jammed=False,
        k=5, used_llm_judge=True, blockers_cleaned=0, ids_restored=0, elapsed_s=1.0, timestamp="t",
    )
    assert "Not a valid target" in r.verdict_label


# ── guaranteed cleanup ──────────────────────────────────────────────────────
def test_run_experiment_cleans_up_on_success(svc, monkeypatch):
    col = FakeCol({"d1", "d2"})
    monkeypatch.setattr(svc, "_exp_collection_obj", lambda: col)
    res = svc.run_experiment(
        "who is huitzilopochtli", "blocker text", method="GCG", style="distraction_nosuffix",
        blocker_origin="pre-made", evaluator=FakeEval(result=sample_eval(), leak_col=col),
    )
    assert res.jammed is True and res.retrieved is True
    assert res.blockers_cleaned == 1   # the tagged blocker was swept
    assert res.ids_restored == 1       # the stray was restored away
    assert col.ids == {"d1", "d2"}     # collection back to the clean corpus


def test_run_experiment_cleans_up_even_when_evaluator_raises(svc, monkeypatch):
    col = FakeCol({"d1", "d2"})
    monkeypatch.setattr(svc, "_exp_collection_obj", lambda: col)
    with pytest.raises(RuntimeError):
        svc.run_experiment(
            "q", "blocker", method="BBO", style="bbo", blocker_origin="fresh",
            evaluator=FakeEval(raise_exc=RuntimeError("ollama down"), leak_col=col),
        )
    assert col.ids == {"d1", "d2"}     # cleanup ran despite the error


# ── export ──────────────────────────────────────────────────────────────────
def test_export_writes_json_and_csv(svc, monkeypatch):
    col = FakeCol({"d1", "d2"})
    monkeypatch.setattr(svc, "_exp_collection_obj", lambda: col)
    res = svc.run_experiment(
        "who is huitzilopochtli", "blocker", method="GCG", style="passive",
        blocker_origin="pre-made", evaluator=FakeEval(result=sample_eval()),
    )
    paths = svc.export(res)
    data = json.loads(open(paths["json"], encoding="utf-8").read())
    assert data["query"] == "who is huitzilopochtli" and data["jammed"] is True
    csv_text = open(paths["csv"], encoding="utf-8").read()
    assert "jammed" in csv_text.splitlines()[0]
    assert "who is huitzilopochtli" in csv_text
