"""Labelled examples: the text to check, plus what the answer should have been.

An ``Example`` is what the frontend's "pick an example from the dataset" control
loads, and it is what makes the oracle comparison possible: it carries gold evidence
keys, so the pipeline can be re-run with retrieval replaced by the annotation.

Gold evidence is stored as ``(doc_id, sent_id)`` keys, never as text. Storing the
text would let an example drift out of alignment with the corpus without anything
noticing; storing keys means ``Corpus.has()`` can prove, at load time, that every
gold key actually exists. ``validate_against`` does exactly that, and you should run
it every time you build a corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from src.core.types import Label

if False:  # typing-only import; keeps this module importable without src.data.corpus
    from src.data.corpus import Corpus


@dataclass(frozen=True, slots=True)
class Example:
    """One labelled item.

    ``text`` is what goes in the response box. For FEVER that is a single
    human-written claim, which is a **named threat to external validity**: FEVER
    claims are mutations of Wikipedia sentences, not LLM output, so a decomposition
    stage has almost nothing to do on them. Do not read a FEVER example's behaviour
    as evidence about behaviour on a real generated paragraph.
    """

    id: str
    text: str
    gold_label: Optional[Label]
    gold_evidence: tuple[tuple[str, int], ...]  # (doc_id, sent_id), possibly empty
    dataset: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_gold_evidence(self) -> bool:
        return len(self.gold_evidence) > 0

    @property
    def gold_evidence_ids(self) -> frozenset[str]:
        """As ``Evidence.id`` strings, for set membership against retrieved results."""
        from src.core.types import evidence_id

        return frozenset(evidence_id(d, s) for d, s in self.gold_evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "gold_label": None if self.gold_label is None else self.gold_label.value,
            "gold_evidence": [[d, s] for d, s in self.gold_evidence],
            "dataset": self.dataset,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        gold = d.get("gold_label")
        return cls(
            id=str(d["id"]),
            text=str(d["text"]),
            gold_label=None if gold is None else Label.parse(gold),
            gold_evidence=tuple((str(x[0]), int(x[1])) for x in (d.get("gold_evidence") or [])),
            dataset=str(d.get("dataset", "unknown")),
            meta=dict(d.get("meta") or {}),
        )


def load_examples(path: str | Path) -> list[Example]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"examples not found: {p}\n"
            "Build them with: python -m scripts.build_debug_corpus --help"
        )
    out: list[Example] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Example.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(f"{p}:{lineno}: {exc}") from None
    return out


def save_examples(examples: Iterable[Example], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return p


def validate_against(examples: Iterable[Example], corpus: "Corpus") -> dict[str, Any]:
    """Prove every gold evidence key resolves in this corpus.

    A dangling gold key means the oracle condition silently retrieves nothing for
    that example, so the retrieved-vs-oracle gap -- the central quantity in the whole
    project -- is computed against an oracle that was itself broken. Loud failure.
    """
    items = list(examples)  # materialise: callers pass generators, and we walk twice
    dangling: list[tuple[str, str, int]] = []
    n_with_gold = 0
    for ex in items:
        if ex.has_gold_evidence:
            n_with_gold += 1
        for doc_id, sent_id in ex.gold_evidence:
            if not corpus.has(doc_id, sent_id):
                dangling.append((ex.id, doc_id, sent_id))
    return {
        "n_examples": len(items),
        "n_with_gold_evidence": n_with_gold,
        "n_dangling_gold_keys": len(dangling),
        "dangling": dangling[:20],
        "ok": not dangling,
    }


def label_counts(examples: Iterable[Example]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        key = "unlabelled" if ex.gold_label is None else ex.gold_label.value
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = ["Example", "load_examples", "save_examples", "validate_against", "label_counts"]
