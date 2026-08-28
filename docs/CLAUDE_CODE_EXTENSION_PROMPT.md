# Claude Code Prompt — Browser Extension + Local Verification Backend

Paste as the opening message. Assumes the pipeline from the previous build exists in `src/`.

---

## 0. What this is and what it is not

Build a Chrome extension (Manifest V3) that flags claim-level hallucinations in LLM responses on chat sites, backed by a FastAPI service running on my own machine.

**This is a demo layer and a data collection instrument. It is not the research contribution.** Timebox it. If you find yourself improving verification quality inside the extension code, stop — that work belongs in `src/`.

Two things the extension must get right, because they are the reason it exists:

1. **It must never display a single "hallucination percentage."** The entire premise of this research is that *contradicted*, *unverifiable*, and *retrieval failed* are different states that naive systems collapse. A scalar collapses them. The badge shows counts by category; the panel shows per-claim verdicts with the evidence that drove each one.
2. **It must let me annotate.** Every claim verdict gets agree/disagree/unclear buttons that write the response text, extracted claims, retrieved evidence, model verdict, and my label to a local corpus. My primary benchmark (FEVER) uses human-written Wikipedia mutations, not LLM output — that is a named threat to external validity in my write-up. This annotation loop is what closes it. Do not treat it as a nice-to-have.

Hardware: RTX 4050 6 GB, 16 GB RAM, Windows 11. Everything runs locally. No data leaves the machine, ever, and the popup must say so in plain language.

---

## 1. Architecture

```
Chat site (claude.ai / chatgpt.com / gemini)
   │  content script: detect completed assistant turn, extract text
   ▼
Background service worker  ──HTTP/SSE──►  FastAPI @ localhost:8000
   │                                          │  reuses src/ pipeline
   │  badge counts                            │  tier 1 → tier 2
   ▼                                          ▼
Inline claim highlights + side panel  ◄──  streamed ClaimVerdicts
```

**Why a local backend and not everything in the extension:** the dense index and the NLI model are gigabytes and need CUDA. There is no version of this that runs in a browser sandbox. State this limitation in the README — it means dev-mode install only, not Chrome Web Store distribution, unless I later ship a packaged installer. Do not pretend otherwise.

---

## 2. Latency strategy — the hard problem

A full pass (decompose → retrieve → rerank → NLI over k evidence, per claim) takes seconds. Users will not wait. Two tiers:

**Tier 1 — fast, runs automatically, target under ~1.5s for a typical response.**
Decompose, run BM25 retrieval only, and compute a cheap *evidence-availability* signal per claim: did retrieval return anything above a floor score. This produces three states only: `likely-checkable`, `no-evidence-found`, `pending`. Badge renders from this. **Tier 1 never asserts that a claim is false** — it only says whether evidence exists to check it. Getting this distinction into the UI is the point.

**Tier 2 — full verification.** Runs on explicit click ("Verify all" or clicking one claim), or in the background at low priority if I enable it in settings. Streams results back per claim via SSE so the panel fills progressively rather than blocking.

Cache aggressively: hash of (claim text, retriever config, verifier config) → verdict, in a local SQLite store. Repeated claims across sessions should be instant.

---

## 3. Extension components

**Site adapters** (`src/extension/adapters/`) — one module per site, each exporting `getAssistantTurns()`, `getTurnText(node)`, and `injectHighlight(node, spans)`. Chat site DOM changes constantly and will break this; isolating it to one small file per site is the whole maintenance strategy. Each adapter needs a selector-health check that logs loudly when it stops matching, rather than silently returning nothing.

**Content script** — MutationObserver watching for a *completed* assistant turn (not a streaming one; detect stream-end per adapter, and debounce). Extract text, send to background, receive verdicts, apply inline highlights over the claim spans.

**Background service worker** — owns the backend connection, the cache, the request queue, and the badge. Must handle the backend being down gracefully: badge shows a neutral "offline" state, never a scary one.

**Inline highlights** — subtle underlines, four states: supported, contradicted, unverifiable, checking. Colors must be distinguishable without relying on hue alone (use underline style too). Hover shows a small card: the top evidence sentence, its source, the entail/contra/neutral split, and the annotation buttons.

**Side panel** — all claims for the current response in a list, each with verdict, confidence, the evidence that drove it, and a plain-English line from `aggregation_trace` saying *why*. Plus a "Verify all" button and an export-this-response button.

**Popup** — backend status, tier-2 auto mode toggle, k slider, model selection, cache size and clear button, annotation count, and the privacy statement.

**Badge** — counts, not a score. Something like a small number with color by worst-category present. If more than one category is present, the badge shows the count of the most severe. Never a percentage.

---

## 4. Backend

FastAPI, reusing the existing pipeline components through the registry — **no duplicated model code**.

- `POST /verify/quick` → tier 1 result for a whole response.
- `POST /verify/full` → SSE stream of `ClaimVerdict` objects as each completes.
- `POST /annotate` → writes an annotation record to the local corpus.
- `GET /health` → model load status, VRAM in use, index status.
- CORS restricted to the specific chat-site origins. Bind to `127.0.0.1` only, never `0.0.0.0`.

Models load once at startup and stay resident. Report peak VRAM at startup and refuse to start with a clear message if it will not fit, rather than OOMing mid-request.

Request queue with a single worker — concurrent requests on a 6 GB card will OOM. Queue depth visible in `/health`.

---

## 5. The annotation corpus — the research payoff

`data/annotations/*.jsonl`, one record per annotated claim:

```
timestamp, site, model_name (if detectable), response_text, claim_text,
claim_span, extractor_name, retrieved_evidence[], system_verdict,
system_confidence, human_label, human_note, config_hash
```

Also record claims where I *disagreed with the decomposition itself* — that is a separate and under-measured error source, and I want it captured, not silently dropped.

Write a small script `scripts/annotation_stats.py` reporting: count by human label, system-vs-human agreement matrix, and the disagreement cases sorted by system confidence. High-confidence disagreements are the most interesting rows in this entire project.

---

## 6. Documentation contract

Same as the rest of the project. Per module: problem, formal I/O, algorithm, walkthrough of only the lines that matter, data flow, how to test it, limitations, rejected alternatives. Append to `docs/DECISIONS.md`.

Extra doc required here — `docs/extension-limitations.md`, written honestly:
- corpus coverage limits what can be verified at all (recent events, personal claims, opinions, reasoning steps are all out of scope — the system must return `unverifiable`, not guess)
- tier 1 is an availability signal, not a truth signal
- DOM adapters are fragile by nature
- latency numbers, measured, on my actual hardware
- the extension's verdicts are not evaluation results and must never be reported as such

---

## 7. Order of work — stop between each

1. Backend endpoints wrapping the existing pipeline, tested with `curl`. **Stop.**
2. Minimal MV3 extension: one adapter, extract text, log to console. No UI. **Stop.**
3. Badge + tier 1. **Stop.**
4. Inline highlights + hover cards + tier 2 streaming. **Stop.**
5. Side panel + annotation buttons + corpus writer. **Stop.**
6. Popup, settings, cache, `annotation_stats.py`.

Start with step 1. Before writing it, tell me which single chat site you would target first and why — I want the one with the most stable DOM, not the one I use most.
