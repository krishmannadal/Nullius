"""Claim extractors: response text -> atomic claims.

Two implementations:

* ``spacy_sentence`` -- sentence split, no decomposition at all.
* ``llm_cached``     -- an LLM decomposition interface backed by a hand-written cache
  file, so it works fully offline with no API key and no local model.

**DECOMPOSITION QUALITY HERE IS ENTIRELY UNMEASURED.** Neither of these has been
evaluated against anything. This matters more than it sounds: decomposition errors do
not stay in the decomposition stage. A sentence split that leaves two facts welded
together ("Curie was born in Warsaw and won two Nobel Prizes") forces the verifier to
return one label for two claims that may have different truth values, and the
resulting error gets attributed to the verifier in every downstream table. An LLM
decomposition that silently drops a clause removes a claim from the denominator
entirely. Both are *upstream* error sources that masquerade as *downstream* ones, and
the frontend's editable-claims panel exists precisely so you can probe this by hand
before committing to an experiment.

THE SPAN CONTRACT
-----------------
``Claim.source_span`` indexes the **original** response string. The obvious way to get
this wrong is to normalise whitespace, split the normalised copy, and report offsets
into it -- the spans then drift, and the frontend highlights the wrong text without
any error. ``SpacySentenceExtractor`` therefore never modifies the string it splits.

``LLMClaimExtractor`` may legitimately produce text that is not a substring of the
response (that is what decomposition *is*), so it reports the span of the sentence a
claim came from, and records the rewrite in ``extractor_meta``. Approximate, and
labelled approximate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.core.config import project_root
from src.core.interfaces import ClaimExtractor
from src.core.registry import register
from src.core.types import Claim, SourceSpan, stable_id


def response_id_for(response: str) -> str:
    """Deterministic id for a response, so the same text yields the same claim ids."""
    return stable_id("resp", response)


# --------------------------------------------------------------------------- #
# sentence split
# --------------------------------------------------------------------------- #

@register("extractor", "spacy_sentence")
class SpacySentenceExtractor(ClaimExtractor):
    """One claim per sentence. No decomposition whatsoever.

    This is the honest floor, not a strawman: it is what most claim-level pipelines
    actually do when they say "atomic claims", and its failure mode -- conjunctions
    and relative clauses surviving intact -- is exactly what an LLM decomposer is
    supposed to fix. Having both behind one interface is what makes that measurable.

    ``model``: ``en_core_web_sm`` (~12 MB) uses a statistical parser and handles
    abbreviations ("Dr.", "U.S.", "Jan.") far better than a rule-based splitter. Set
    ``use_sentencizer=True`` to fall back to spaCy's rule-based ``sentencizer``, which
    needs no model download but breaks on abbreviations -- and those breaks would show
    up downstream as decomposition errors, contaminating the very thing you would be
    trying to measure.
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        use_sentencizer: bool = False,
        min_chars: int = 3,
    ) -> None:
        super().__init__()
        import spacy

        self.model = model
        self.use_sentencizer = bool(use_sentencizer)
        self.min_chars = int(min_chars)
        if use_sentencizer:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
        else:
            try:
                # Parser only: NER and tagging cost time and we use neither.
                self.nlp = spacy.load(model, disable=["ner", "lemmatizer", "attribute_ruler"])
            except OSError as exc:
                raise OSError(
                    f"spaCy model {model!r} is not installed. Run:\n"
                    f"    python -m spacy download {model}\n"
                    f"or set use_sentencizer: true to use the rule-based fallback "
                    f"(no download, worse on abbreviations).\nOriginal error: {exc}"
                ) from None

    def extract(self, response: str) -> list[Claim]:
        # NOTE: `response` is passed through unmodified. Normalising it here and
        # reporting spans into the normalised copy is the classic way to make the
        # frontend's highlighting silently wrong.
        doc = self.nlp(response)
        rid = response_id_for(response)
        claims: list[Claim] = []
        for i, sent in enumerate(doc.sents):
            text = sent.text.strip()
            if len(text) < self.min_chars:
                continue
            # sent.start_char/end_char index the ORIGINAL string; re-derive the exact
            # span of the stripped text so span.text_from(response) == claim.text.
            start = sent.start_char + (len(sent.text) - len(sent.text.lstrip()))
            claims.append(
                Claim.new(
                    response_id=rid,
                    text=text,
                    extractor_name=self.name,
                    source_span=SourceSpan(start, start + len(text)),
                    extractor_meta={"sentence_index": i, "decomposed": False},
                )
            )
        return claims


# --------------------------------------------------------------------------- #
# LLM decomposition, cached
# --------------------------------------------------------------------------- #

DEFAULT_CACHE = "data/debug/llm_claims_cache.json"


@register("extractor", "llm_cached")
class LLMClaimExtractor(ClaimExtractor):
    """LLM-style decomposition served from a hand-written cache. Offline by default.

    The point of this class is the *interface*, not the model. It defines how a
    decomposer is called and how its output is recorded, so that swapping in a real
    API call or a local model later is a one-class change rather than a redesign.

    **Cache-first, and by default cache-only.** The cache is a plain JSON file mapping
    a hash of the response text to a list of claim strings, so you can write
    decompositions by hand -- including deliberately bad ones -- and see exactly how
    they propagate. With ``on_miss="split"`` (the default) an unseen response falls
    back to sentence splitting and marks itself as a fallback in ``extractor_meta``;
    with ``on_miss="error"`` it raises instead, which is what you want in an experiment
    where a silent fallback would quietly mix two extractors in one table.

    No network call is implemented. ``on_miss="call"`` raises ``NotImplementedError``
    naming what would have to be built, rather than pretending an API exists.
    """

    def __init__(
        self,
        cache_path: str = DEFAULT_CACHE,
        on_miss: str = "split",          # "split" | "error" | "call"
        fallback_model: str = "en_core_web_sm",
        use_sentencizer: bool = False,
    ) -> None:
        super().__init__()
        if on_miss not in ("split", "error", "call"):
            raise ValueError(f"on_miss must be 'split', 'error' or 'call', got {on_miss!r}")
        self.on_miss = on_miss
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_absolute():
            self.cache_path = project_root() / cache_path
        self.cache: dict[str, list[str]] = self._load_cache()
        self._fallback: SpacySentenceExtractor | None = None
        self._fallback_args = (fallback_model, use_sentencizer)

    def _load_cache(self) -> dict[str, list[str]]:
        if not self.cache_path.is_file():
            return {}
        raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        # Accept both {"entries": {...}} and a bare mapping, so a hand-written file
        # does not need boilerplate.
        entries = raw.get("entries", raw) if isinstance(raw, dict) else {}
        return {str(k): [str(c) for c in v] for k, v in entries.items()}

    @staticmethod
    def cache_key(response: str) -> str:
        """Hash of the exact response text. Whitespace-sensitive on purpose: two
        responses differing only in whitespace are different inputs to a decomposer."""
        return hashlib.blake2b(response.encode("utf-8"), digest_size=16).hexdigest()[:16]

    def _fallback_extractor(self) -> SpacySentenceExtractor:
        if self._fallback is None:
            model, use_sentencizer = self._fallback_args
            self._fallback = SpacySentenceExtractor(model=model, use_sentencizer=use_sentencizer)
        return self._fallback

    def extract(self, response: str) -> list[Claim]:
        key = self.cache_key(response)
        rid = response_id_for(response)

        if key not in self.cache:
            if self.on_miss == "error":
                raise KeyError(
                    f"no cached decomposition for response {key!r} "
                    f"(cache: {self.cache_path}). Add one, or set on_miss='split'."
                )
            if self.on_miss == "call":
                raise NotImplementedError(
                    "on_miss='call' would need: a provider client, a decomposition "
                    "prompt, response parsing, retry/timeout handling, and a cache "
                    "write-back. None of that exists yet, and this build is offline "
                    "by design. Write the decomposition into the cache file instead."
                )
            claims = self._fallback_extractor().extract(response)
            return [
                Claim.new(
                    response_id=rid,
                    text=c.text,
                    extractor_name=self.name,
                    source_span=c.source_span,
                    extractor_meta={
                        **c.extractor_meta,
                        "cache_key": key,
                        "cache_hit": False,
                        "fallback": "sentence_split",
                        "decomposed": False,
                    },
                )
                for c in claims
            ]

        texts = self.cache[key]
        # Span attribution is APPROXIMATE by construction: a decomposed claim is
        # usually not a substring of the response. We report the span of the first
        # sentence that shares the most words, and record that it is a guess.
        claims: list[Claim] = []
        seen: set[str] = set()
        for i, text in enumerate(texts):
            text = text.strip()
            if not text or text in seen:
                continue          # de-duplicate: LLM decomposers repeat themselves
            seen.add(text)
            span = self._locate(response, text)
            claims.append(
                Claim.new(
                    response_id=rid,
                    text=text,
                    extractor_name=self.name,
                    source_span=span,
                    extractor_meta={
                        "cache_key": key,
                        "cache_hit": True,
                        "claim_index": i,
                        "decomposed": True,
                        "span_is_exact": span is not None and span.text_from(response) == text,
                        "n_claims_in_cache": len(texts),
                    },
                )
            )
        return claims

    @staticmethod
    def _locate(response: str, text: str) -> SourceSpan | None:
        """Best-effort span for a rewritten claim. Exact match first, then word overlap."""
        idx = response.find(text)
        if idx >= 0:
            return SourceSpan(idx, idx + len(text))
        words = {w.lower().strip(".,;:!?") for w in text.split() if len(w) > 3}
        if not words:
            return None
        best_span, best_overlap = None, 0
        cursor = 0
        for chunk in response.split("."):
            start = response.find(chunk, cursor)
            if start < 0:
                continue
            cursor = start + len(chunk)
            chunk_words = {w.lower().strip(".,;:!?") for w in chunk.split() if len(w) > 3}
            overlap = len(words & chunk_words)
            if overlap > best_overlap:
                best_overlap, best_span = overlap, SourceSpan(start, start + len(chunk))
        return best_span


__all__ = ["DEFAULT_CACHE", "LLMClaimExtractor", "SpacySentenceExtractor", "response_id_for"]
