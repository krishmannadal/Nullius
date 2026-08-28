"""Contract tests for src/core/registry.py.

The registry's job is to fail loudly on config mistakes.  Every test here is a
config mistake that would otherwise change results silently.
"""

from __future__ import annotations

import pytest

from src.core import registry
from src.core.interfaces import Aggregator, ClaimExtractor, Reranker, Retriever, Verifier
from src.core.types import Claim, ClaimVerdict, Evidence, EvidenceVerdict, Label


# --------------------------------------------------------------------------- #
# fixtures: minimal legal components, registered into a scratch registry
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def scratch_registry():
    """Each test gets an empty registry; builtins are not imported."""
    saved = {kind: dict(reg) for kind, reg in registry._REGISTRY.items()}
    registry._reset_for_tests()
    yield
    for kind, reg in saved.items():
        registry._REGISTRY[kind] = reg


class _Extractor(ClaimExtractor):
    def __init__(self, mode: str = "sentence") -> None:
        super().__init__()
        self.mode = mode

    def extract(self, response: str) -> list[Claim]:
        return [Claim.new("r", response, self.name)]


class _Retriever(Retriever):
    def __init__(self, corpus: str, k_default: int = 5) -> None:
        super().__init__()
        self.corpus = corpus
        self.k_default = k_default

    def retrieve(self, claim: Claim, k: int) -> list[Evidence]:
        return []


class _Reranker(Reranker):
    def rerank(self, claim: Claim, ev: list[Evidence], k: int) -> list[Evidence]:
        return ev[:k]


class _Verifier(Verifier):
    def score(self, claim: Claim, ev: Evidence) -> EvidenceVerdict:
        return EvidenceVerdict(claim.id, ev.id, None, None, None, 0.0, self.name, 0.0)


class _Aggregator(Aggregator):
    def aggregate(self, claim: Claim, v: list[EvidenceVerdict]) -> ClaimVerdict:
        return ClaimVerdict(
            claim.id,
            Label.INSUFFICIENT,
            0.0,
            False,
            tuple(v),
            self.name,
            {"rule": "stub", "explanation": "stub", "decisive_evidence_ids": []},
        )


def register_all() -> None:
    registry.register("extractor", "stub")(_Extractor)
    registry.register("retriever", "stub")(_Retriever)
    registry.register("reranker", "noop")(_Reranker)
    registry.register("verifier", "stub")(_Verifier)
    registry.register("aggregator", "stub")(_Aggregator)


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #

def test_register_sets_the_name_used_in_traces():
    registry.register("retriever", "stub")(_Retriever)
    obj = registry.build("retriever", {"name": "stub", "params": {"corpus": "debug"}})
    assert obj.name == "stub"
    assert obj.describe()["params"] == {"corpus": "debug"}


def test_registering_under_the_wrong_kind_fails_at_decoration():
    with pytest.raises(registry.RegistryError, match="does not subclass"):
        registry.register("verifier", "oops")(_Retriever)


def test_duplicate_name_with_a_different_class_is_rejected():
    registry.register("retriever", "stub")(_Retriever)

    class _Other(Retriever):
        def retrieve(self, claim, k):
            return []

    with pytest.raises(registry.RegistryError, match="already registered"):
        registry.register("retriever", "stub")(_Other)


def test_re_registering_the_same_class_is_a_no_op():
    registry.register("retriever", "stub")(_Retriever)
    registry.register("retriever", "stub")(_Retriever)  # module reload under pytest
    assert registry.available("retriever") == ["stub"]


def test_available_is_what_the_ui_dropdown_reads():
    register_all()
    assert registry.available_all() == {
        "extractor": ["stub"],
        "retriever": ["stub"],
        "reranker": ["noop"],
        "verifier": ["stub"],
        "aggregator": ["stub"],
    }


# --------------------------------------------------------------------------- #
# lookup and construction
# --------------------------------------------------------------------------- #

def test_unknown_name_error_lists_the_valid_names():
    registry.register("retriever", "stub")(_Retriever)
    with pytest.raises(registry.RegistryError, match=r"registered: \['stub'\]"):
        registry.get("retriever", "bm25")


def test_unknown_param_is_rejected_not_ignored():
    """`top_k: 10` where the constructor says `k_default` must not pass silently."""
    registry.register("retriever", "stub")(_Retriever)
    with pytest.raises(registry.RegistryError, match="unknown params"):
        registry.build("retriever", {"name": "stub", "params": {"corpus": "d", "top_k": 10}})


def test_missing_required_param_is_reported():
    registry.register("retriever", "stub")(_Retriever)
    with pytest.raises(registry.RegistryError, match="missing required params"):
        registry.build("retriever", {"name": "stub"})


def test_shorthand_string_spec_is_accepted():
    registry.register("extractor", "stub")(_Extractor)
    obj = registry.build("extractor", "stub")
    assert isinstance(obj, _Extractor) and obj.mode == "sentence"


def test_spec_without_a_name_is_rejected():
    with pytest.raises(registry.RegistryError, match="has no 'name' key"):
        registry.normalise_spec({"params": {}})


# --------------------------------------------------------------------------- #
# pipeline assembly
# --------------------------------------------------------------------------- #

BASE_CFG = {
    "pipeline": {"k": 3},
    "components": {
        "extractor": "stub",
        "retriever": {"name": "stub", "params": {"corpus": "debug"}},
        "verifier": "stub",
        "aggregator": "stub",
    },
}


def test_build_pipeline_defaults_the_reranker_to_noop():
    register_all()
    pipe = registry.build_pipeline(BASE_CFG)
    assert pipe.reranker.name == "noop"
    assert pipe.k == 3


def test_build_pipeline_reports_missing_components():
    register_all()
    cfg = {"pipeline": {"k": 3}, "components": {"extractor": "stub"}}
    with pytest.raises(registry.RegistryError, match="missing components"):
        registry.build_pipeline(cfg)


def test_build_pipeline_rejects_an_unknown_component_kind():
    register_all()
    cfg = {"components": dict(BASE_CFG["components"], fuser="stub")}
    with pytest.raises(registry.RegistryError, match="unknown component kinds"):
        registry.build_pipeline(cfg)


def test_pipeline_describe_is_json_shaped_provenance():
    register_all()
    described = registry.build_pipeline(BASE_CFG).describe()
    assert described["retriever"]["name"] == "stub"
    assert described["retriever"]["params"] == {"corpus": "debug"}
    assert described["k"] == 3
