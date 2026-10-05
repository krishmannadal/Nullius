"""Build the FEVER-derived debug corpus.

    python -m scripts.build_debug_corpus --n-claims 200 --n-docs 4000

READ THIS BEFORE BELIEVING ANY NUMBER PRODUCED OVER THE RESULT.

The corpus this builds contains, by construction, the gold evidence page for every
claim it ships with. Its distractor pool is not adversarial. Retrieval recall over it
is therefore an artifact of how it was assembled, not a property of a retriever, and
every downstream number inherits that. It exists so that all five pipeline stages can
be exercised on real FEVER data at a size that fits on a laptop. It is a debug
harness. It is not an evaluation setup, and the config it writes says so.

THE ALIGNMENT PROBLEM THIS SCRIPT EXISTS TO GET RIGHT
-----------------------------------------------------
FEVER gold evidence is ``(page, sentence_id)`` into the June-2017 Wikipedia snapshot
shipped as ``wiki-pages.zip``. Each page record carries a ``lines`` field::

    "0\\tMarie Curie was a physicist.\\tMarie Curie\\n1\\tShe was born in Warsaw.\\n2\\t"

i.e. newline-separated records of ``{index}\\t{sentence}\\t{hyperlink annotations...}``.

Two traps, both silent:

1. **Empty sentences are real and they occupy an index.** Record ``2`` above has empty
   text. Filtering empties out shifts every later sentence down by one, so gold
   evidence ``(page, 7)`` silently resolves to what was sentence 8. Nothing crashes.
   The claim gets checked against the wrong sentence and the error looks like a bad
   NLI model.
2. **The declared index is authoritative, not the position in the file.** So this
   script places each sentence at ``sentences[declared_index]``, padding gaps with
   empty strings, and asserts the result. It never uses ``enumerate``.

``Corpus.fingerprint`` and ``validate_against`` catch a *missing* key. Neither can
catch a *shifted* one -- a shifted key still resolves, just to the wrong text. That is
why the parse is index-driven and why ``tests/test_fever_parse.py`` checks it directly
against hand-written fixtures.

WHAT COUNTS AS A DISTRACTOR
---------------------------
Two pools, in priority order:

1. Gold pages of *other* dev claims not in our sample. These are substantive articles
   that some annotator considered evidence-worthy, so they make a more realistic pool
   than uniform random sampling, which mostly yields short stubs. They are also not
   selected by any retriever, so no circularity is introduced.
2. Uniform random pages, to fill the remainder.

Consequence worth stating: a pool built this way contains pages that may genuinely
bear on our claims without being annotated gold for them. FEVER's annotations are
known to be incomplete in this way, so ``is_gold == False`` means "not annotated for
this claim", never "irrelevant".
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows consoles default to cp1252, and Wikipedia page titles are full of
# characters it cannot encode (combining accents, CJK, ...). Without this, the
# script completes 5.4 M pages of work and then dies in a print() statement.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
        pass

from src.core.types import Label
from src.data.corpus import Corpus, Document
from src.data.examples import Example, label_counts, save_examples, validate_against

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# FEVER wiki `lines` parsing -- the alignment-critical part
# --------------------------------------------------------------------------- #

class ParseStats:
    def __init__(self) -> None:
        self.pages = 0
        self.empty_lines_field = 0
        self.non_integer_index = 0
        self.duplicate_index = 0
        self.padded_gaps = 0
        self.empty_sentences = 0
        self.total_sentences = 0

    def as_dict(self) -> dict[str, int]:
        return {k: v for k, v in vars(self).items()}


def parse_lines_field(raw: str, stats: ParseStats | None = None) -> tuple[str, ...]:
    """FEVER ``lines`` -> a tuple where ``result[i]`` is FEVER's sentence *i*.

    Index-driven, never positional. Gaps are padded with "" so that
    ``sentences[sent_id]`` is exactly the sentence FEVER's gold evidence names.
    """
    if not raw:
        if stats:
            stats.empty_lines_field += 1
        return ()

    by_index: dict[int, str] = {}
    for record in raw.split("\n"):
        if not record.strip():
            continue
        fields = record.split("\t")
        try:
            idx = int(fields[0])
        except (ValueError, IndexError):
            # Real occurrence in the dump: a continuation fragment with no index.
            # Dropping it is safe *because* we index by declared id -- it cannot
            # shift anything.
            if stats:
                stats.non_integer_index += 1
            continue
        text = fields[1] if len(fields) > 1 else ""
        if idx in by_index and stats:
            stats.duplicate_index += 1
        by_index[idx] = text

    if not by_index:
        return ()

    size = max(by_index) + 1
    out = tuple(by_index.get(i, "") for i in range(size))
    if stats:
        stats.padded_gaps += size - len(by_index)
        stats.empty_sentences += sum(1 for s in out if not s.strip())
        stats.total_sentences += size
    return out


def normalize_page_id(page_id: str) -> str:
    """NFC-normalise a Wikipedia page title. Apply at EVERY boundary where one enters.

    MEASURED BUG IN FEVER ITSELF, not a hypothetical. The two FEVER files disagree
    about Unicode normalisation for the same title::

        shared_task_dev.jsonl:  'Cléopâtre'   (NFD, decomposed)
        wiki-pages.zip:         'Cléopâtre'          (NFC, precomposed)

    They render identically and compare unequal. Without normalisation, exact string
    matching silently drops the page and the claim gets dropped with it.

    Scale, measured over all 2,892 dev gold pages:
        32 are non-NFC  =  1.1% of all gold pages
                        = 72.7% of the 44 NON-ASCII gold pages

    That 1% aggregate figure is the dangerous part: the loss is not random, it is
    concentrated almost entirely on non-English entity names (Bjork, Curacao, Cafe
    Society, Cleopatre, ...). Any per-slice analysis of non-ASCII entities would be
    computed on a sample with three quarters of it missing, and the aggregate would
    look fine.

    NFC rather than NFD because the wiki dump -- the larger artifact and the one that
    defines ``doc_id`` -- is already NFC, so this changes 32 titles rather than
    5.4 million.
    """
    return unicodedata.normalize("NFC", page_id)


def is_real_wiki_member(name: str) -> bool:
    """Filter zip members down to the 109 actual data files.

    wiki-pages.zip was built on macOS and ships an AppleDouble resource fork beside
    every data file: ``__MACOSX/wiki-pages/._wiki-001.jsonl``. Those end in ``.jsonl``
    but are small binary blobs, and ``._`` sorts before ``w``, so a naive
    ``endswith(".jsonl")`` filter hits one *first* and dies on JSONDecodeError at
    char 0. 218 members, 109 of them real.
    """
    if name.startswith(("__MACOSX/", "._")) or "/._" in name:
        return False
    return name.endswith(".jsonl")


def iter_wiki_pages(zip_path: Path) -> Iterator[tuple[str, str]]:
    """Stream (page_id, lines_field) out of wiki-pages.zip without extracting it.

    The zip is ~1.7 GB and the extracted form is ~4 GB; we need a few thousand pages
    out of ~5.4 M, so streaming keeps peak disk at just the zip.
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(n for n in zf.namelist() if is_real_wiki_member(n))
        if not members:
            raise SystemExit(f"{zip_path} contains no .jsonl members: {zf.namelist()[:5]}")
        for member in members:
            with zf.open(member) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    page_id = obj.get("id")
                    if page_id:
                        yield normalize_page_id(page_id), obj.get("lines", "")


# --------------------------------------------------------------------------- #
# FEVER dev claims
# --------------------------------------------------------------------------- #

def load_dev(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evidence_groups(row: dict[str, Any]) -> list[list[tuple[str, int]]]:
    """FEVER evidence -> list of alternative groups of (page, sent_id).

    Each group is a *complete* evidence set on its own; a group with >1 key is
    multi-hop. NEI rows carry ``[[id, null, null, null]]`` and yield nothing.
    """
    groups: list[list[tuple[str, int]]] = []
    for group in row.get("evidence") or []:
        keys = [
            (normalize_page_id(e[2]), int(e[3]))
            for e in group
            if e[2] is not None and e[3] is not None
        ]
        if keys:
            groups.append(keys)
    return groups


def sample_claims(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Stratified sample, as equal per label as `n` allows, deterministic under `rng`.

    The remainder of `n // n_labels` is distributed over the first labels in sorted
    order, so `--n-claims 200` yields exactly 200 (67/67/66) rather than 198. Which
    class gets the extra is fixed by label sort order, not by chance, so the sample
    stays reproducible.
    """
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    labels = sorted(by_label)
    base, remainder = divmod(n, len(labels))
    picked: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        want = base + (1 if i < remainder else 0)
        pool = sorted(by_label[label], key=lambda r: r["id"])  # stable before sampling
        picked.extend(rng.sample(pool, min(want, len(pool))))
    picked.sort(key=lambda r: r["id"])
    return picked


def claim_kind(row: dict[str, Any], groups: list[list[tuple[str, int]]]) -> str:
    """A coarse slice tag for the UI. Hand heuristic, not a taxonomy."""
    if not groups:
        return "no-annotated-evidence"
    if any(len({p for p, _ in g}) > 1 for g in groups):
        return "multi-hop"
    text = row["claim"]
    if any(ch.isdigit() for ch in text):
        return "numeric-or-date"
    return "single-page"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, default=ROOT / "data/raw/shared_task_dev.jsonl")
    ap.add_argument("--wiki", type=Path, default=ROOT / "data/raw/wiki-pages.zip")
    ap.add_argument("--out", type=Path, default=ROOT / "data/debug")
    ap.add_argument("--n-claims", type=int, default=200)
    ap.add_argument("--n-docs", type=int, default=4000, help="total pages in the corpus")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    for p in (args.dev, args.wiki):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}\nSee docs/data-fever.md for the download commands.")

    rng = random.Random(args.seed)

    # ---- 1. pick claims, work out which pages we must have -------------------
    dev = load_dev(args.dev)
    picked = sample_claims(dev, args.n_claims, rng)
    picked_ids = {r["id"] for r in picked}

    required_pages: set[str] = set()
    for row in picked:
        for group in evidence_groups(row):
            required_pages.update(p for p, _ in group)

    other_gold_pages: set[str] = set()
    for row in dev:
        if row["id"] in picked_ids:
            continue
        for group in evidence_groups(row):
            other_gold_pages.update(p for p, _ in group)
    other_gold_pages -= required_pages

    print(f"claims sampled     {len(picked)}  ({dict(Counter(r['label'] for r in picked))})")
    print(f"required gold pages{len(required_pages):>5}")
    print(f"other-dev-gold pool{len(other_gold_pages):>5}")

    # ---- 2. one streaming pass over the 1.7 GB zip ---------------------------
    n_fill = max(0, args.n_docs - len(required_pages) - len(other_gold_pages))
    stats = ParseStats()
    kept: dict[str, tuple[str, ...]] = {}
    others: dict[str, tuple[str, ...]] = {}
    reservoir: list[tuple[str, tuple[str, ...]]] = []
    seen_random = 0

    print(f"scanning {args.wiki.name} (one pass; reservoir target {n_fill}) ...")
    for i, (page_id, raw_lines) in enumerate(iter_wiki_pages(args.wiki), start=1):
        stats.pages += 1
        if i % 1_000_000 == 0:
            print(f"  ... {i:,} pages scanned, {len(kept)}/{len(required_pages)} gold found")

        if page_id in required_pages:
            kept[page_id] = parse_lines_field(raw_lines, stats)
            continue
        if page_id in other_gold_pages:
            others[page_id] = parse_lines_field(raw_lines, stats)
            continue
        if n_fill:
            # Reservoir sampling: uniform over all non-gold pages in ONE pass.
            seen_random += 1
            if len(reservoir) < n_fill:
                reservoir.append((page_id, parse_lines_field(raw_lines, stats)))
            else:
                j = rng.randrange(seen_random)
                if j < n_fill:
                    reservoir[j] = (page_id, parse_lines_field(raw_lines, stats))

    missing = sorted(required_pages - set(kept))
    if missing:
        print(f"\nWARNING: {len(missing)} gold pages absent from the dump, e.g. {missing[:5]}")

    # ---- 3. assemble, in a deterministic order -------------------------------
    docs: list[Document] = []
    for page_id in sorted(kept):
        docs.append(Document(page_id, page_id.replace("_", " "), kept[page_id]))
    for page_id in sorted(others):
        docs.append(Document(page_id, page_id.replace("_", " "), others[page_id]))
    for page_id, sents in sorted(reservoir):
        docs.append(Document(page_id, page_id.replace("_", " "), sents))
    docs = [d for d in docs if d.sentences]  # a page with no parsable lines is not evidence
    corpus = Corpus(docs)

    # ---- 4. examples, dropping claims whose gold did not survive -------------
    examples: list[Example] = []
    dropped: list[str] = []
    for row in picked:
        groups = evidence_groups(row)
        gold = sorted({key for g in groups for key in g})
        resolvable = [k for k in gold if corpus.has(*k)]
        if gold and len(resolvable) != len(gold):
            dropped.append(f"{row['id']} ({len(gold) - len(resolvable)} of {len(gold)} keys unresolvable)")
            continue
        examples.append(
            Example(
                id=f"fever-dev-{row['id']}",
                text=row["claim"],
                gold_label=Label.parse(row["label"]),
                gold_evidence=tuple(resolvable),
                dataset="fever-dev",
                meta={
                    "fever_id": row["id"],
                    "verifiable": row.get("verifiable"),
                    "kind": claim_kind(row, groups),
                    # The grouped structure is preserved because a group is a
                    # COMPLETE alternative evidence set. Flattening to `gold_evidence`
                    # is right for Recall@k and wrong for the official FEVER score,
                    # which needs "did we recover at least one full group".
                    "evidence_groups": [[[p, s] for p, s in g] for g in groups],
                    "n_evidence_groups": len(groups),
                },
            )
        )

    report = validate_against(examples, corpus)
    if not report["ok"]:
        raise SystemExit(f"BUILD ERROR: dangling gold keys after assembly: {report['dangling']}")

    # A key can resolve and still be wrong if the parse shifted indices. Gold
    # sentences are near-verbatim sources for FEVER claims, so an EMPTY gold
    # sentence is a strong misparse signal.
    empty_gold = [
        (ex.id, key) for ex in examples for key in ex.gold_evidence
        if not corpus.get(*key).text.strip()
    ]
    if empty_gold:
        print(f"\nWARNING: {len(empty_gold)} gold keys point at empty sentences "
              f"(possible misparse), e.g. {empty_gold[:3]}")

    # ---- 5. write ------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    corpus.to_jsonl(args.out / "corpus.jsonl")
    save_examples(examples, args.out / "examples.jsonl")
    manifest = {
        "built_by": "scripts/build_debug_corpus.py",
        "harness_kind": "debug",
        "warning": (
            "Contains the gold evidence page for every example by construction. "
            "Distractors are not adversarial. Recall over this corpus is an artifact "
            "of its assembly, not a property of a retriever."
        ),
        "seed": args.seed,
        "n_claims_requested": args.n_claims,
        "n_docs_requested": args.n_docs,
        "source_dev": str(args.dev.name),
        "source_wiki": str(args.wiki.name),
        "corpus": corpus.stats(),
        "n_examples": len(examples),
        "labels": label_counts(examples),
        "kinds": dict(Counter(e.meta["kind"] for e in examples)),
        "dropped_claims": dropped,
        "n_gold_pages": len(kept),
        "n_other_gold_pages": len(others),
        "n_random_pages": len(reservoir),
        "n_empty_gold_sentences": len(empty_gold),
        "parse_stats": stats.as_dict(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nwrote {args.out}")
    for key in ("n_docs", "n_sentences", "fingerprint"):
        print(f"  {key:<26} {corpus.stats()[key]}")
    print(f"  {'n_examples':<26} {len(examples)}")
    print(f"  {'labels':<26} {label_counts(examples)}")
    print(f"  {'kinds':<26} {manifest['kinds']}")
    print(f"  {'dropped claims':<26} {len(dropped)}")
    print(f"  {'gold/other-gold/random':<26} {len(kept)}/{len(others)}/{len(reservoir)}")
    print(f"  {'empty gold sentences':<26} {len(empty_gold)}")
    print(f"  parse stats: {stats.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
