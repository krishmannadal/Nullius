"""The corpus: documents, sentences, and the addressing scheme everything else trusts.

One idea does all the work here: **a corpus has a canonical sentence order**, and
that order is the only thing that maps a FAISS row back to a piece of text. If the
order changes and an index does not, every dense retrieval result silently points at
the wrong sentence. Nothing crashes. Recall@k stays plausible. The error analysis
becomes fiction.

So the corpus computes a ``fingerprint`` -- a hash over its ordered
``(doc_id, sent_id)`` keys -- and any artifact built from it (the FAISS index, a
BM25 pickle) records that fingerprint and refuses to load against a corpus that does
not match. That check is the reason this module exists as more than a JSON reader.

On-disk format, one JSON object per line::

    {"doc_id": "Marie_Curie", "title": "Marie Curie", "sentences": ["...", "..."]}

``sent_id`` is the 0-based index into ``sentences``, which is exactly what
``src.core.types.evidence_id(doc_id, sent_id)`` addresses. There is no second
sentence-numbering scheme anywhere in the project.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

_KEY_SEP = b"\x1f"


@dataclass(frozen=True, slots=True)
class CorpusSentence:
    """One addressable unit of evidence."""

    doc_id: str
    sent_id: int
    text: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.sent_id)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    sentences: tuple[str, ...]


class Corpus:
    """An ordered, addressable collection of sentences.

    Row order is **load order**: documents in file order, sentences in document
    order. It is deliberately not sorted -- sorting would silently reorder rows when
    a doc_id changes case or gains a diacritic, which is the exact class of change
    that invalidates an index without looking like it should.
    """

    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents: tuple[Document, ...] = tuple(documents)

        seen_docs: set[str] = set()
        rows: list[CorpusSentence] = []
        for doc in self.documents:
            if doc.doc_id in seen_docs:
                raise ValueError(f"duplicate doc_id in corpus: {doc.doc_id!r}")
            seen_docs.add(doc.doc_id)
            for sent_id, text in enumerate(doc.sentences):
                rows.append(CorpusSentence(doc.doc_id, sent_id, text))

        self.sentences: tuple[CorpusSentence, ...] = tuple(rows)
        # row  -> key  is self.sentences[row].key
        # key  -> row  is this dict. Both directions are needed and both are tested.
        self._row_by_key: dict[tuple[str, int], int] = {
            s.key: row for row, s in enumerate(self.sentences)
        }
        self._doc_by_id: dict[str, Document] = {d.doc_id: d for d in self.documents}
        self._fingerprint: str | None = None
        self._content_fingerprint: str | None = None

    # ----------------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self.sentences)

    def __iter__(self) -> Iterator[CorpusSentence]:
        return iter(self.sentences)

    @property
    def n_docs(self) -> int:
        return len(self.documents)

    def row(self, doc_id: str, sent_id: int) -> int:
        """key -> row. Raises rather than returning -1: a missing key is a bug."""
        try:
            return self._row_by_key[(doc_id, sent_id)]
        except KeyError:
            raise KeyError(f"({doc_id!r}, {sent_id}) is not in this corpus") from None

    def at(self, row: int) -> CorpusSentence:
        """row -> sentence. The inverse of `row()`, and the FAISS lookup path."""
        return self.sentences[row]

    def text_at(self, row: int) -> str:
        return self.sentences[row].text

    def get(self, doc_id: str, sent_id: int) -> CorpusSentence:
        return self.sentences[self.row(doc_id, sent_id)]

    def has(self, doc_id: str, sent_id: int) -> bool:
        return (doc_id, sent_id) in self._row_by_key

    def document(self, doc_id: str) -> Document:
        return self._doc_by_id[doc_id]

    def texts(self) -> list[str]:
        """All sentence texts in row order. This is what BM25 and the encoder consume."""
        return [s.text for s in self.sentences]

    # ------------------------------------------------------------ fingerprint

    def fingerprint(self) -> str:
        """Hash of the ordered (doc_id, sent_id) keys.

        Covers *addressing*, not content: changing a sentence's wording does not
        change the fingerprint, but inserting, deleting, or reordering one does.
        That is the right sensitivity -- row->key alignment is what an index depends
        on. (Content drift is caught separately, by the corpus file's own hash in the
        run manifest.)
        """
        if self._fingerprint is None:
            h = hashlib.blake2b(digest_size=16)
            for s in self.sentences:
                h.update(s.doc_id.encode("utf-8"))
                h.update(_KEY_SEP)
                h.update(str(s.sent_id).encode("ascii"))
                h.update(_KEY_SEP)
            self._fingerprint = h.hexdigest()[:16]
        return self._fingerprint

    def content_fingerprint(self) -> str:
        """Hash of the ordered (doc_id, sent_id, text) tuples.

        Covers the actual wording of every sentence in the corpus.
        """
        if getattr(self, "_content_fingerprint", None) is None:
            h = hashlib.blake2b(digest_size=32)
            for s in self.sentences:
                h.update(s.doc_id.encode("utf-8"))
                h.update(_KEY_SEP)
                h.update(str(s.sent_id).encode("ascii"))
                h.update(_KEY_SEP)
                h.update(s.text.encode("utf-8"))
                h.update(_KEY_SEP)
            self._content_fingerprint = h.hexdigest()[:16]
        return self._content_fingerprint

    # ------------------------------------------------------------------- I/O

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Corpus:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"corpus not found: {p}\n"
                "Build one with: python -m scripts.build_debug_corpus --help"
            )
        docs: list[Document] = []
        with p.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{p}:{lineno}: not valid JSON: {exc}") from None
                try:
                    sentences = tuple(str(s) for s in obj["sentences"])
                    docs.append(
                        Document(
                            doc_id=str(obj["doc_id"]),
                            title=str(obj.get("title", obj["doc_id"])),
                            sentences=sentences,
                        )
                    )
                except KeyError as exc:
                    raise ValueError(f"{p}:{lineno}: missing key {exc}") from None
        return cls(docs)

    def to_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for doc in self.documents:
                fh.write(
                    json.dumps(
                        {"doc_id": doc.doc_id, "title": doc.title, "sentences": list(doc.sentences)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return p

    def stats(self) -> dict[str, object]:
        lengths = [len(d.sentences) for d in self.documents]
        return {
            "n_docs": self.n_docs,
            "n_sentences": len(self.sentences),
            "sentences_per_doc_mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "sentences_per_doc_min": min(lengths) if lengths else 0,
            "sentences_per_doc_max": max(lengths) if lengths else 0,
            "fingerprint": self.fingerprint(),
            "content_fingerprint": self.content_fingerprint(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Corpus docs={self.n_docs} sentences={len(self.sentences)} fp={self.fingerprint()}>"


__all__ = ["Corpus", "CorpusSentence", "Document"]
