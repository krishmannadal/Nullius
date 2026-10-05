"""Retrievers: BM25, dense (bge-small + FAISS flat), and RRF hybrid.

All three are placeholders with defensible defaults. Nothing here is tuned, and the
debug corpus makes any recall number computed over it uninformative.

The two things worth reading closely:

* **Index alignment.** The dense index is a FAISS matrix whose row *i* means "the
  i-th sentence in the corpus's canonical order". If the corpus is rebuilt with one
  extra sentence and the index is not, every result silently points at the wrong
  text. So the index is written with a manifest recording
  ``Corpus.fingerprint()``, the model name and the dimension, and loading refuses on
  any mismatch. See ``_load_or_build_index``.
* **Fusion by rank, not by score.** ``HybridRetriever`` uses reciprocal rank fusion.
  The reason is in ``HybridRetriever``'s docstring; the short version is that BM25
  scores are unbounded and query-dependent, so any per-query normalisation makes the
  fused result depend on the score *spread* of the candidate list rather than on
  agreement between the arms.

Heavy imports (torch, sentence_transformers, faiss) happen inside ``DenseRetriever``,
never at module import. The registry imports this module during ``load_builtins()``,
and a BM25-only run on a machine without CUDA must not pay for that.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.core.config import project_root
from src.core.interfaces import Retriever
from src.core.registry import build as build_component
from src.core.registry import get as get_component
from src.core.registry import register
from src.core.types import Claim, Evidence
from src.data.corpus import Corpus

# --------------------------------------------------------------------------- #
# shared corpus loading
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=8)
def _load_corpus_version(path: str, digest: str) -> Corpus:
    """One Corpus object per path per process.

    A HybridRetriever builds a BM25 arm and a dense arm, and both need the corpus.
    Loading it twice would double the memory and -- much worse -- allow the two arms
    to disagree about row order if the file changed between loads.
    """
    return Corpus.from_jsonl(path)


def load_corpus_cached(path: str) -> Corpus:
    resolved = str(Path(path).resolve())
    return _load_corpus_version(resolved, hashlib.sha256(Path(resolved).read_bytes()).hexdigest())


load_corpus_cached.cache_clear = _load_corpus_version.cache_clear


def _accepts_param(cls: type, name: str) -> bool:
    """Does this class's __init__ take `name` (or **kwargs)?"""
    import inspect

    sig = inspect.signature(cls.__init__)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return name in sig.parameters


def _resolve(path: str | Path) -> Path:
    """Config paths are repo-relative; absolute paths pass through unchanged."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics. No stemming, no stopword removal.

    Stated so it can be argued with rather than discovered:
      * **No stopword list** -- BM25's IDF term already discounts frequent words, and
        a hand-picked stoplist is an untracked hyperparameter.
      * **No stemming** -- "Curie" and "Curies" stay distinct. On numeric and
        entity-heavy claims (the slice this project cares about) stemming helps
        rarely and merges distinct entities occasionally.
      * **Digits kept as tokens** -- "1898" must be matchable, because date and
        numeric claims are a named failure slice.
    """
    return _TOKEN_RE.findall(text.lower())


@register("retriever", "bm25")
class BM25Retriever(Retriever):
    """Okapi BM25 over corpus sentences.

    Scores are raw BM25 and are NOT comparable across queries (they depend on the
    query's IDF mass). That is why the UI shows them in their own column and why
    fusion uses ranks.
    """

    def __init__(
        self,
        corpus_path: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        super().__init__()
        from rank_bm25 import BM25Okapi  # local: keeps the dependency off the import path

        self.corpus_path = str(corpus_path)
        self.corpus = load_corpus_cached(str(_resolve(corpus_path)))
        self.k1 = float(k1)
        self.b = float(b)
        # Row order here IS the corpus row order; scores[i] refers to corpus.at(i).
        self._tokenized = [bm25_tokenize(s.text) for s in self.corpus.sentences]
        self._bm25 = BM25Okapi(self._tokenized, k1=self.k1, b=self.b)

    def retrieve(self, claim: Claim, k: int) -> list[Evidence]:
        scores = self._bm25.get_scores(bm25_tokenize(claim.text))
        # argsort descending, stable on ties by row index so results are deterministic
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        out: list[Evidence] = []
        for rank, row in enumerate(order, start=1):
            sent = self.corpus.at(row)
            out.append(
                Evidence.new(
                    doc_id=sent.doc_id,
                    sent_id=sent.sent_id,
                    text=sent.text,
                    score=float(scores[row]),
                    retriever_name=self.name,
                    rank=rank,
                    retriever_meta={"bm25_score": float(scores[row]), "bm25_rank": rank},
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Dense
# --------------------------------------------------------------------------- #


@register("retriever", "dense")
class DenseRetriever(Retriever):
    """Bi-encoder + FAISS flat inner-product index.

    Embeddings are L2-normalised, so inner product == cosine similarity and
    ``IndexFlatIP`` is an exact cosine search. Flat (not IVF/PQ) because the debug
    corpus is a few thousand sentences: exact search costs microseconds and removes
    approximate-search recall loss as a confound. At full-corpus scale this choice
    changes; that is a separate, costed decision.
    """

    def __init__(
        self,
        corpus_path: str,
        model_name: str = "BAAI/bge-small-en-v1.5",
        index_dir: str = "data/debug/index",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        batch_size: int = 64,
        device: str | None = None,
        rebuild: bool = False,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        self.corpus_path = str(corpus_path)
        self.corpus = load_corpus_cached(str(_resolve(corpus_path)))
        self.model_name = model_name
        self.index_dir = _resolve(index_dir)
        # BGE models are trained with an asymmetric retrieval instruction on the QUERY
        # side only; passages are embedded bare. Prefixing both sides, or neither,
        # measurably changes retrieval and is a common silent misuse of this family.
        self.query_prefix = query_prefix
        self.batch_size = int(batch_size)
        self.rebuild = bool(rebuild)

        from sentence_transformers import SentenceTransformer  # local import: heavy

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.encoder = SentenceTransformer(model_name, device=device, revision=revision)
        model_config = getattr(getattr(self.encoder[0], "auto_model", None), "config", None)
        self.model_identity = {"model": model_name, "requested_revision": revision,
                               "resolved_revision": getattr(model_config, "_commit_hash", None),
                               "device": device}
        self.dim = int(self.encoder.get_sentence_embedding_dimension())
        self.embedding_identity = {
            "model": model_name, "revision": self.model_identity["resolved_revision"] or revision,
            "max_seq_length": self.encoder.max_seq_length, "dimension": self.dim,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "normalize_embeddings": True,
        }
        self.index = self._load_or_build_index()

    # ------------------------------------------------------------------ index

    def _index_paths(self) -> tuple[Path, Path]:
        slug = self.model_name.replace("/", "__")
        identity_hash = hashlib.sha256(json.dumps(self.embedding_identity, sort_keys=True).encode()).hexdigest()[:12]
        stem = f"{slug}-{self.corpus.fingerprint()}-{identity_hash}"
        return (self.index_dir / f"{stem}.faiss", self.index_dir / f"{stem}.manifest.json")

    def _load_or_build_index(self):
        import faiss  # local import: heavy
        import numpy as np

        index_path, manifest_path = self._index_paths()
        if index_path.is_file() and manifest_path.is_file() and not self.rebuild:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._check_manifest(manifest)
            index = faiss.read_index(str(index_path))
            if index.ntotal != len(self.corpus):
                raise RuntimeError(
                    f"index at {index_path} has {index.ntotal} vectors but the corpus has "
                    f"{len(self.corpus)} sentences. Delete the index and rebuild."
                )
            return index

        vectors = self.encoder.encode(
            self.corpus.texts(),          # row order == corpus row order. Load-bearing.
            batch_size=self.batch_size,
            normalize_embeddings=True,    # makes inner product == cosine
            convert_to_numpy=True,
            show_progress_bar=len(self.corpus) > 5000,
        ).astype(np.float32)              # FAISS requires float32; float64 silently fails

        index = faiss.IndexFlatIP(self.dim)
        index.add(vectors)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        manifest_path.write_text(
            json.dumps(
                {
                    "corpus_fingerprint": self.corpus.fingerprint(),
                    "corpus_content_fingerprint": self.corpus.content_fingerprint(),
                    "corpus_path": self.corpus_path,
                    "n_sentences": len(self.corpus),
                    "model_name": self.model_name,
                    "embedding_identity": self.embedding_identity,
                    "dim": self.dim,
                    "normalized": True,
                    "index_type": "IndexFlatIP",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return index

    def _check_manifest(self, manifest: dict[str, Any]) -> None:
        """Refuse a stale index rather than returning confidently wrong sentences."""
        checks = [
            ("embedding_identity", self.embedding_identity),
            ("corpus_fingerprint", self.corpus.fingerprint()),
            ("corpus_content_fingerprint", self.corpus.content_fingerprint()),
            ("model_name", self.model_name),
            ("dim", self.dim),
            ("n_sentences", len(self.corpus)),
        ]
        for key, expected in checks:
            got = manifest.get(key)
            if got != expected:
                raise RuntimeError(
                    f"stale FAISS index: manifest {key}={got!r} but current value is "
                    f"{expected!r}. The index does not describe this corpus/model; "
                    "rebuild it (rebuild: true, or delete the .faiss and .manifest.json)."
                )

    # --------------------------------------------------------------- retrieve

    def retrieve(self, claim: Claim, k: int) -> list[Evidence]:
        import numpy as np

        query = self.encoder.encode(
            [self.query_prefix + claim.text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        k_eff = min(k, len(self.corpus))
        scores, rows = self.index.search(query, k_eff)

        out: list[Evidence] = []
        for rank, (row, score) in enumerate(zip(rows[0].tolist(), scores[0].tolist()), start=1):
            if row < 0:  # FAISS pads with -1 when fewer than k results exist
                continue
            sent = self.corpus.at(row)
            out.append(
                Evidence.new(
                    doc_id=sent.doc_id,
                    sent_id=sent.sent_id,
                    text=sent.text,
                    score=float(score),
                    retriever_name=self.name,
                    rank=rank,
                    retriever_meta={"dense_score": float(score), "dense_rank": rank},
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Hybrid (RRF)
# --------------------------------------------------------------------------- #

DEFAULT_RRF_K = 60


@register("retriever", "hybrid")
class HybridRetriever(Retriever):
    r"""Reciprocal rank fusion over two or more retrievers.

    .. math::
        \mathrm{RRF}(d) = \sum_{a \in \text{arms}} \frac{1}{K + r_a(d)}

    where :math:`r_a(d)` is *d*'s 1-based rank in arm *a*, and *d* absent from an arm
    contributes nothing. :math:`K = 60` is the constant from Cormack, Clarke &
    Buettcher (2009), where it was found insensitive across TREC collections. It is a
    **placeholder here, not a tuned value** -- its only role is to flatten the
    difference between ranks 1 and 2 relative to the difference between ranks 1 and
    20, so a single arm's confident top hit cannot dominate a document that both arms
    rank moderately well.

    **Why RRF rather than normalised score fusion.** BM25 scores are unbounded and
    depend on the query's IDF mass; cosine scores are bounded in [-1, 1]. To add them
    you must normalise, and the only normalisation available at query time is over
    the candidate list itself (min-max or z-score). That makes the fused score depend
    on the *spread* of the candidates: a query where every candidate is irrelevant has
    its best candidate rescaled to 1.0 exactly as a query with a perfect match does.
    Rank-based fusion is invariant to that. The cost, stated: RRF discards magnitude,
    so it cannot distinguish "clearly the best" from "barely the best", and it cannot
    express "both arms scored everything terribly". Recovering that is what
    ``retriever_meta`` carries the per-arm scores for.

    Ties in the fused score are broken by best per-arm rank, then by evidence id, so
    the output is deterministic.
    """

    def __init__(
        self,
        corpus_path: str,
        arms: Sequence[Any] | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        candidates_per_arm: int = 50,
    ) -> None:
        super().__init__()
        self.corpus_path = str(corpus_path)
        self.rrf_k = int(rrf_k)
        self.candidates_per_arm = int(candidates_per_arm)
        if self.rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")

        specs = list(arms) if arms else [{"name": "bm25"}, {"name": "dense"}]
        self.arms: list[Retriever] = []
        for spec in specs:
            spec = dict(spec) if isinstance(spec, dict) else {"name": str(spec)}
            params = dict(spec.get("params") or {})
            # Ergonomics: an arm that takes a corpus_path inherits ours, so the path
            # is written once in the config. Arms that take no corpus (a stub, or a
            # future API-backed retriever) are left alone -- injecting unconditionally
            # would make this class impossible to fuse anything but our own retrievers
            # with. The actual safety property is the fingerprint check below.
            if _accepts_param(get_component("retriever", spec["name"]), "corpus_path"):
                params.setdefault("corpus_path", self.corpus_path)
            spec["params"] = params
            self.arms.append(build_component("retriever", spec))  # type: ignore[arg-type]

        # Fusing two arms over different corpora would silently produce evidence ids
        # from incompatible sentence numbering: same id string, different sentence.
        # RRF would then "agree" about a document neither arm actually returned.
        fingerprints = {
            (arm.corpus.fingerprint(), arm.corpus.content_fingerprint()): type(arm).__name__
            for arm in self.arms
            if getattr(arm, "corpus", None) is not None
        }
        if len(fingerprints) > 1:
            raise ValueError(
                "hybrid arms are built over different corpora "
                f"(fingerprints: {fingerprints}); their evidence ids are not comparable"
            )

    def retrieve(self, claim: Claim, k: int) -> list[Evidence]:
        fused: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        meta: dict[str, dict[str, Any]] = {}
        exemplar: dict[str, Evidence] = {}

        for arm in self.arms:
            hits = arm.retrieve(claim, self.candidates_per_arm)
            for hit in hits:
                # De-duplication across arms relies on Evidence.id being corpus
                # identity only (ADR-003). If id included the retriever, the same
                # sentence would be fused with itself and double-counted.
                fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (self.rrf_k + hit.rank)
                best_rank[hit.id] = min(best_rank.get(hit.id, 10**9), hit.rank)
                meta.setdefault(hit.id, {}).update(hit.retriever_meta)
                exemplar.setdefault(hit.id, hit)

        order = sorted(fused, key=lambda eid: (-fused[eid], best_rank[eid], eid))[:k]
        out: list[Evidence] = []
        for rank, eid in enumerate(order, start=1):
            base = exemplar[eid]
            detail = dict(meta[eid])
            detail["rrf_score"] = fused[eid]
            detail["fused_rank"] = rank
            detail["n_arms_hit"] = sum(
                1 for key in ("bm25_rank", "dense_rank") if key in detail
            )
            out.append(
                Evidence.new(
                    doc_id=base.doc_id,
                    sent_id=base.sent_id,
                    text=base.text,
                    score=fused[eid],
                    retriever_name=self.name,
                    rank=rank,
                    retriever_meta=detail,
                )
            )
        return out


__all__ = [
    "DEFAULT_RRF_K",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "bm25_tokenize",
    "load_corpus_cached",
]
