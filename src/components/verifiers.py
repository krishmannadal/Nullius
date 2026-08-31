"""Verifiers: score exactly ONE (claim, evidence) pair.

Three implementations. Two are real signals, one is a null baseline whose whole
purpose is to fail:

* ``nli_deberta``   -- cross-encoder NLI, the entail/contra/neutral distribution.
* ``similarity``    -- cosine between bi-encoder embeddings, threshold-free.
* ``claim_only``    -- **null baseline.** Runs the NLI model with an EMPTY premise, so
  it cannot read the evidence. If this scores near the real pipeline later, the
  benchmark has claim-side artifacts and every downstream number is suspect.

TWO MEASURED FOOTGUNS THIS MODULE EXISTS TO DEFUSE
--------------------------------------------------

**1. The label order is not what you would guess, and it is not constant across
checkpoints.** For ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``, measured::

    id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

Many MNLI checkpoints ship the *reverse* (``0: contradiction ... 2: entailment``).
Hardcoding index 0 as "contradiction" would silently swap Supported and Contradicted
for every claim, and the numbers would still look like plausible model output. So the
mapping is **read from ``model.config.id2label`` at load time**, never written down,
and ``_resolve_label_indices`` fails loudly if the three names are not all present.

**2. ``tokenizer.model_max_length`` is 1000000000000000019884624838656.** That is not
a joke -- it is the sentinel this tokenizer ships when no limit is configured.
Consequence: ``truncation=True`` **does nothing at all** unless an explicit
``max_length`` is passed alongside it. A long evidence sentence then produces a
sequence longer than the position embeddings and either errors deep in the model or
silently degrades. Every call here passes ``max_length`` explicitly.

Related: ``truncation="longest_first"`` rather than the default ``"only_first"``
behaviour people assume. With ``truncation_side="right"`` on a (claim, evidence)
pair, naive truncation eats the END of the pair -- the evidence -- but if the claim
were the longer member it would be silently cut instead. ``longest_first`` removes
tokens from whichever member is longer, one at a time, which is the only policy that
never destroys a short claim.
"""

from __future__ import annotations

import time

from src.core.interfaces import Verifier
from src.core.registry import register
from src.core.types import Claim, Evidence, EvidenceVerdict

#: Passed explicitly on every tokenizer call. See module docstring footgun #2.
DEFAULT_MAX_LENGTH = 256


def make_verdict(
    claim: Claim,
    ev: Evidence,
    *,
    verifier_name: str,
    latency_ms: float,
    p_entail: float | None = None,
    p_contra: float | None = None,
    p_neutral: float | None = None,
    similarity: float | None = None,
) -> EvidenceVerdict:
    """Build a verdict, copying the retrieval signal off the Evidence.

    ``evidence_rank`` / ``evidence_score`` are denormalised here because an
    Aggregator only ever sees verdicts. Doing it in one helper means all three
    verifiers do it identically -- if each did it by hand, one of them would
    eventually forget and ``WeightedByRetrievalScore`` would silently fall back to
    uniform weights for that verifier only.
    """
    return EvidenceVerdict(
        claim_id=claim.id,
        evidence_id=ev.id,
        p_entail=p_entail,
        p_contra=p_contra,
        p_neutral=p_neutral,
        similarity=similarity,
        verifier_name=verifier_name,
        latency_ms=latency_ms,
        evidence_rank=ev.rank,
        evidence_score=ev.score,
    )


def _resolve_label_indices(id2label: dict[int, str]) -> dict[str, int]:
    """model.config.id2label -> {'entailment': i, 'neutral': j, 'contradiction': k}.

    Loud on anything unexpected. A checkpoint whose labels are LABEL_0/LABEL_1/... has
    no usable mapping, and guessing one is exactly the failure this function exists to
    prevent.
    """
    lowered = {int(i): str(name).strip().lower() for i, name in id2label.items()}
    wanted = ("entailment", "neutral", "contradiction")
    out: dict[str, int] = {}
    for name in wanted:
        matches = [i for i, got in lowered.items() if got == name]
        if len(matches) != 1:
            raise ValueError(
                f"cannot resolve NLI label {name!r} from id2label={id2label}. "
                "This checkpoint does not expose a usable 3-way mapping; refusing to "
                "guess, because a wrong guess swaps Supported and Contradicted "
                "silently."
            )
        out[name] = matches[0]
    return out


# --------------------------------------------------------------------------- #
# NLI
# --------------------------------------------------------------------------- #

@register("verifier", "nli_deberta")
class NLIVerifier(Verifier):
    """Cross-encoder NLI over (evidence as premise, claim as hypothesis).

    Direction matters and is fixed here: **evidence is the premise, the claim is the
    hypothesis**. "Does the evidence entail the claim?" is the question; the reverse
    ("does the claim entail the evidence?") is a different and mostly meaningless one
    for fact verification, and getting it backwards produces a model that looks merely
    bad rather than misconfigured.

    Measured on an RTX 4050 at fp16, checkpoint DeBERTa-v3-base-mnli-fever-anli:
    359 MiB of weights, 467 MiB peak at batch 16 x 512 tokens, ~50-95 ms per forward.
    VRAM is not a constraint here; it was worth measuring rather than assuming.
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str | None = None,
        fp16: bool = True,
    ) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.max_length = int(max_length)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # fp16 on CPU is slow and partly unimplemented; silently ignore the request.
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=self.dtype)
            .to(device)
            .eval()
        )
        # THE mapping. Read, never assumed. See module docstring footgun #1.
        self.label_index = _resolve_label_indices(dict(self.model.config.id2label))
        self._torch = torch

    def _score_pair(self, premise: str, hypothesis: str) -> tuple[float, float, float, float]:
        torch = self._torch
        t0 = time.perf_counter()
        enc = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation="longest_first",   # never silently destroy the shorter member
            max_length=self.max_length,   # REQUIRED: model_max_length is ~1e19 here
            padding=False,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        # softmax in fp32: fp16 softmax over 3 logits can round a near-1.0 probability
        # to exactly 1.0 and the others to 0.0, which then fails the simplex check by
        # more than tolerance in the other direction.
        probs = logits.float().softmax(-1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return (
            float(probs[self.label_index["entailment"]]),
            float(probs[self.label_index["contradiction"]]),
            float(probs[self.label_index["neutral"]]),
            latency_ms,
        )

    def score(self, claim: Claim, ev: Evidence) -> EvidenceVerdict:
        p_e, p_c, p_n, latency = self._score_pair(ev.text, claim.text)
        # Renormalise: three fp32 floats from a softmax can sum to 1 +/- 1e-7, which is
        # inside PROB_SUM_TOL, but the cast to float() above can drift further on some
        # backends. Cheap insurance, and it makes the stored numbers exact.
        total = p_e + p_c + p_n
        if total > 0:
            p_e, p_c, p_n = p_e / total, p_c / total, p_n / total
        return make_verdict(
            claim, ev,
            verifier_name=self.name, latency_ms=latency,
            p_entail=p_e, p_contra=p_c, p_neutral=p_n,
        )


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

@register("verifier", "similarity")
class SimilarityVerifier(Verifier):
    """Cosine similarity between claim and evidence embeddings. Threshold-free.

    Emits ``similarity`` and leaves the three NLI probabilities as ``None`` -- see
    ``EvidenceVerdict``'s all-or-nothing rule. Writing 0.0 into them instead would make
    a similarity verdict indistinguishable from a confidently-neutral NLI verdict in
    every aggregate.

    No threshold is applied, deliberately. "Is 0.62 similar enough?" is an aggregator
    decision that must appear in ``aggregation_trace``, not a constant buried here.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        query_prefix: str = "",
    ) -> None:
        super().__init__()
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        # Empty by default: this is a SYMMETRIC claim-vs-evidence comparison, not the
        # asymmetric query-vs-passage task the BGE retrieval prefix is for. Using the
        # retrieval prefix here would measure something subtly different from what the
        # DenseRetriever measures, while looking identical.
        self.query_prefix = query_prefix
        self.encoder = SentenceTransformer(model_name, device=device)

    def score(self, claim: Claim, ev: Evidence) -> EvidenceVerdict:
        t0 = time.perf_counter()
        vecs = self.encoder.encode(
            [self.query_prefix + claim.text, ev.text],
            normalize_embeddings=True,   # so the dot product below IS cosine
            convert_to_numpy=True,
        )
        cos = float(vecs[0] @ vecs[1])
        # Clamp: normalisation leaves |cos| at 1 + ~1e-7, and EvidenceVerdict validates
        # [-1, 1] strictly. Clamping is right; widening the contract would not be.
        cos = max(-1.0, min(1.0, cos))
        return make_verdict(
            claim, ev,
            verifier_name=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            similarity=cos,
        )


# --------------------------------------------------------------------------- #
# Null baseline
# --------------------------------------------------------------------------- #

@register("verifier", "claim_only")
class ClaimOnlyVerifier(Verifier):
    """NULL BASELINE. Scores the claim with the evidence removed.

    It still takes ``ev`` and still returns a verdict keyed to it, so it runs on
    exactly the same code path as the real verifier and every aggregator works
    unchanged. Only the *premise* is blanked.

    **What a good score here means.** If this reaches anywhere near the evidence-using
    pipeline on a benchmark, the model is exploiting claim-side artifacts -- surface
    cues in how claims were written -- rather than reading evidence. On FEVER
    specifically this is a live risk, because REFUTES claims are human-authored
    mutations and mutation style is learnable. Wire it in from the start, run it first,
    and treat a strong result as an alarm rather than a success.

    ``premise`` is configurable so you can separate two different nulls: the empty
    string (no premise at all) versus a fixed irrelevant premise (a premise exists but
    says nothing). They can differ, and the difference is informative.
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str | None = None,
        fp16: bool = True,
        premise: str = "",
    ) -> None:
        super().__init__()
        self.premise = premise
        # Composition rather than inheritance: this shares the NLI machinery but is
        # emphatically NOT a kind of NLIVerifier, and should never be substitutable for
        # one by accident.
        self._nli = NLIVerifier(
            model_name=model_name, max_length=max_length, device=device, fp16=fp16
        )
        self.model_name = model_name
        self.max_length = int(max_length)
        self.device = self._nli.device
        self.label_index = self._nli.label_index

    def score(self, claim: Claim, ev: Evidence) -> EvidenceVerdict:
        # ev.text is deliberately unused. That is the entire point of this class.
        p_e, p_c, p_n, latency = self._nli._score_pair(self.premise, claim.text)
        total = p_e + p_c + p_n
        if total > 0:
            p_e, p_c, p_n = p_e / total, p_c / total, p_n / total
        return make_verdict(
            claim, ev,
            verifier_name=self.name, latency_ms=latency,
            p_entail=p_e, p_contra=p_c, p_neutral=p_n,
        )


__all__ = [
    "DEFAULT_MAX_LENGTH",
    "ClaimOnlyVerifier",
    "NLIVerifier",
    "SimilarityVerifier",
    "make_verdict",
]
