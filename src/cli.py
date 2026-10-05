"""Command line entry point.

    python -m src.cli list-components
    python -m src.cli run     --config configs/mini.yaml --limit 20
    python -m src.cli analyze --config configs/mini.yaml --text "Marie Curie was born in Paris."
    python -m src.cli show    results/<run_id>/traces.jsonl

Design rule: **the CLI contains no pipeline logic.** It parses arguments, builds
components through the registry, calls ``src.pipeline``, and prints. Anything it did
that the API could not would be a divergence between what you debug here and what the
frontend runs — so it does nothing.

``--set a.b=c`` applies the same override machinery the Streamlit sidebar will use, so a
config reached from the command line and the same config reached from the UI resolve
identically and therefore hash identically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import (
    apply_overrides,
    configure_console,
    load_config,
    set_global_seed,
)
from src.core.registry import available_all, build_pipeline
from src.core.trace_io import TraceWriter, read_traces, run_dir_for, write_run
from src.data.corpus import Corpus
from src.data.examples import load_examples
from src.data.metrics import recall_at_k, summarise_run
from src.pipeline import RunContext, analyze_example


def _resolve_path(cfg: dict, key: str) -> Path:
    from src.core.config import project_root

    raw = Path(cfg["paths"][key])
    return raw if raw.is_absolute() else project_root() / raw


def _load(args) -> dict:
    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, args.set)
        cfg["_config_path"] = str(Path(args.config).resolve())
    set_global_seed(int(cfg.get("seed", 0)))
    return cfg


def _banner(cfg: dict) -> None:
    kind = (cfg.get("harness") or {}).get("kind", "unknown")
    note = (cfg.get("harness") or {}).get("corpus_note", "")
    if kind == "debug":
        print("=" * 78)
        print("  DEBUG HARNESS -- NOT AN EVALUATION SETUP")
        print(f"  {note.strip()}")
        print("  Numbers produced here describe the harness, not a method.")
        print("=" * 78)


# --------------------------------------------------------------------------- #

def cmd_list_components(args) -> int:
    for kind, names in available_all().items():
        print(f"{kind:<12} {', '.join(names) if names else '(none registered)'}")
    return 0


def cmd_analyze(args) -> int:
    import asyncio

    from src.service import NulliusService

    cfg = _load(args)
    _banner(cfg)
    service = NulliusService(default_config_path=args.config)
    trace = asyncio.run(
        service.analyze(
            args.text,
            overrides=args.set,
            retrieve_k=args.retrieve_k,
        )
    )

    print(f"\nrun_id      {trace.run_id}")
    print(f"config_hash {trace.config_hash}   git {(trace.git_sha or 'none')[:8]}")
    print(f"timings     {trace.timings}")
    for claim in trace.claims:
        verdict = trace.verdict_by_claim(claim.id)
        print(f"\n  CLAIM  {claim.text}")
        print(f"  ->     {verdict.label.value}  conf={verdict.confidence:.3f}"
              f"{'  ABSTAINED' if verdict.abstained else ''}")
        print(f"  why:   {verdict.aggregation_trace['explanation']}")
        for ev in trace.evidence_by_claim[claim.id]:
            pv = next((p for p in verdict.per_evidence if p.evidence_id == ev.id), None)
            probs = (f"e={pv.p_entail:.2f} c={pv.p_contra:.2f} n={pv.p_neutral:.2f}"
                     if pv and pv.has_nli else (f"sim={pv.similarity:.3f}" if pv else "-"))
            print(f"    {ev.rank}. {probs}  {ev.doc_id[:32]:<32} {ev.text[:60]}")
    if args.json:
        print(trace.to_json_line())
    return 0


def cmd_run(args) -> int:
    cfg = _load(args)
    _banner(cfg)
    pipe = build_pipeline(cfg)
    ctx = RunContext.create(cfg, pipe)

    corpus = Corpus.from_jsonl(_resolve_path(cfg, "corpus"))
    examples = load_examples(_resolve_path(cfg, "examples"))
    if args.limit:
        examples = examples[: args.limit]
    by_id = {e.id: e for e in examples}

    out_dir = run_dir_for(ctx.run_id, cfg["paths"].get("results_dir", "results"))
    print(f"\ncorpus   {corpus.n_docs} docs / {len(corpus)} sentences / fp {corpus.fingerprint()}")
    print(f"examples {len(examples)}")
    print(f"run_id   {ctx.run_id}   config_hash {ctx.config_hash}")
    print(f"out      {out_dir}\n")

    retrieved_traces, oracle_by_example = [], {}
    with TraceWriter(out_dir / "traces.jsonl") as tw, TraceWriter(out_dir / "oracle.jsonl") as ow:
        for i, ex in enumerate(examples, start=1):
            r, o = analyze_example(
                pipe, ex, corpus, ctx,
                retrieve_k=args.retrieve_k,
                check_contracts=not args.no_checks,
            )
            tw.write(r)
            retrieved_traces.append(r)
            if o is not None:
                ow.write(o)
                oracle_by_example[ex.id] = o

            verdict = r.verdicts[0] if r.verdicts else None
            gold = ex.gold_label.value if ex.gold_label else "?"
            got = verdict.label.value if verdict else "?"
            mark = "ok " if got == gold else "XX "
            rec = recall_at_k(next(iter(r.evidence_by_claim.values()), ()), ex.gold_evidence_ids)
            o_got = (oracle_by_example[ex.id].verdicts[0].label.value
                     if ex.id in oracle_by_example and oracle_by_example[ex.id].verdicts else "-")
            print(f"  [{i:>3}/{len(examples)}] {mark}{ex.id:<20} gold={gold:<13} "
                  f"got={got:<13} oracle={o_got:<13} R@k={'n/a' if rec is None else f'{rec:.2f}'}"
                  f"  {r.timings['total_ms']:.0f} ms")

    metrics = summarise_run(retrieved_traces, oracle_by_example, by_id)
    write_run(
        out_dir,
        resolved_config=ctx.resolved_config,
        metrics=metrics,
        harness_kind=(cfg.get("harness") or {}).get("kind", "debug"),
    )

    print(f"\nwrote {out_dir}")
    print(f"  traces.jsonl      {len(retrieved_traces)}")
    print(f"  oracle.jsonl      {len(oracle_by_example)}")
    print(f"  predicted         {metrics['predicted_labels']}")
    print(f"  gold              {metrics['gold_labels']}")
    print(f"  abstained         {metrics['n_abstained']}")
    agree = metrics["agreement_with_gold"]
    print(f"  agreement w/gold  {'n/a' if agree is None else f'{agree:.3f}'}  "
          f"(NOT an evaluation number -- see metrics.json _warning)")
    rec = metrics["recall_at_k_mean"]
    print(f"  mean Recall@k     {'n/a' if rec is None else f'{rec:.3f}'} "
          f"over {metrics['n_recall_defined']} claims with gold evidence")
    gap = metrics["retrieval_attributable_error"]
    print(f"  --- retrieval-attributable error ({gap['n_claims_with_both_conditions']} claims "
          f"with both conditions) ---")
    for key in ("retrieved_agreement", "oracle_agreement", "gap"):
        v = gap[key]
        print(f"    {key:<22} {'n/a' if v is None else f'{v:+.3f}' if key == 'gap' else f'{v:.3f}'}")
    print(f"    fixed by oracle        {gap['n_fixed_by_oracle']}")
    print(f"    broken by oracle       {gap['n_broken_by_oracle']}")
    print(f"  timings (ms)      {metrics['timings_ms_total']}")
    return 0


def cmd_show(args) -> int:
    traces = list(read_traces(args.path))
    print(f"{len(traces)} traces in {args.path}")
    for t in traces[: args.limit]:
        for claim in t.claims:
            v = t.verdict_by_claim(claim.id)
            print(f"\n[{t.mode}] {claim.text[:70]}")
            print(f"   {v.label.value} conf={v.confidence:.3f} "
                  f"({v.aggregator_name}) {'ABSTAINED' if v.abstained else ''}")
            print(f"   {v.aggregation_trace['explanation'][:200]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_console()
    ap = argparse.ArgumentParser(prog="src.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-components", help="what the registry knows about")
    p.set_defaults(func=cmd_list_components)
    for name, fn, extra in (("run", cmd_run, True), ("analyze", cmd_analyze, False)):
        q = sub.add_parser(name)
        q.add_argument("--config", default="configs/mini.yaml")
        q.add_argument("--set", action="append", default=[], metavar="a.b=c",
                       help="override a config value; same parser as the YAML itself")
        q.add_argument("--retrieve-k", type=int, default=None,
                       help="candidate pool size handed to the reranker (default: k)")
        if extra:
            q.add_argument("--limit", type=int, default=20)
            q.add_argument("--no-checks", action="store_true",
                           help="skip stage-boundary contract checks (do not)")
        else:
            q.add_argument("--text", required=True)
            q.add_argument("--json", action="store_true", help="also print the raw trace")
        q.set_defaults(func=fn)

    q = sub.add_parser("show", help="render saved traces without any model stack")
    q.add_argument("path")
    q.add_argument("--limit", type=int, default=5)
    q.set_defaults(func=cmd_show)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
