# `src/core/config.py` — config resolution and provenance

## 1. Problem

One rule: *a result that cannot be regenerated from its config does not exist.*
Three things break it in practice.

* A number that affects a result lives in code rather than in YAML, so it is not in
  the hash and not in the trace.
* The hash is taken over the *file*, so a CLI override or a sidebar change produces a
  different run under the same hash — two different results, one identifier.
* Provenance is faked when unavailable (a placeholder `git_sha`, a made-up
  timestamp), which is worse than absent because it looks trustworthy.

## 2. Formal I/O

```
load_config(path)             : Path → dict          # + "_config_path" (absolute, machine-local)
apply_overrides(cfg, ["a.b=v"]) : (Mapping, Iterable[str]) → dict   # deep-copied, never mutates
canonical_json(cfg)           : Mapping → str        # sorted keys, no spaces, "_"-keys dropped
config_hash(cfg, length=12)   : Mapping → str        # blake2b-128 hex, truncated
git_sha() / git_is_dirty()    : () → Optional[str] / Optional[bool]  # None outside a repo
set_global_seed(seed)         : int → None           # random, PYTHONHASHSEED, numpy, torch
provenance(cfg)               : Mapping → {config_hash, git_sha, git_dirty, config_path}
```

Hash definition:

$$\texttt{config\_hash} = \mathrm{blake2b}_{128}\big(\texttt{json}(\{k{:}v \in \texttt{cfg} \mid k_0 \ne \texttt{"\_"}\},\ \text{sorted},\ \text{compact})\big)[:12]$$

Truncating to 12 hex digits = 48 bits. With a few thousand configs the birthday
collision probability is ~10⁻⁸ — negligible for a run identifier, and it stays short
enough to read in a directory name.

## 3. Algorithm

```
apply_overrides(cfg, items):
    out ← json.loads(json.dumps(cfg))          # deep copy AND a JSON-safety check
    for "dotted.path=raw" in items:
        walk/create dicts along path[:-1]
        out[path[-1]] ← yaml.safe_load(raw)    # same scalar rules as the file itself
    return out

config_hash(cfg):
    drop keys starting with "_"                # machine-local paths must not enter the hash
    json.dumps(sort_keys, separators=(",",":"), ensure_ascii=False)
    blake2b(digest_size=16).hexdigest()[:12]
```

## 4. Code walkthrough — the lines that matter

**Overrides parse as YAML scalars (`config.py:63`).**
```python
node[keys[-1]] = yaml.safe_load(raw)
```
`k=5` becomes `int`, `name=bm25` becomes `str`, `use_gpu=true` becomes `bool`. The
point is not convenience: it means a value typed on the command line and the same
value written into the YAML resolve to the *same Python object* and therefore hash
identically. Without this, `--set pipeline.k=5` and `k: 5` would be `"5"` and `5`,
two hashes for one experiment.

**The deep copy that is also a validation (`config.py:50`).**
```python
out = json.loads(json.dumps({k: v for k, v in cfg.items()}))
```
Round-tripping through JSON copies the config *and* proves it is JSON-safe, which is
required later because the whole resolved config is embedded in every `Trace`. A
non-serialisable value fails here, at config time, rather than after a GPU run.

**Underscore keys excluded from the hash (`config.py:74`).**
```python
clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
```
`_config_path` is `C:\Users\...` on your laptop and `/home/...` anywhere else. In the
hash, the same experiment would get two identities. It is still recorded in
`provenance()` — just not in the thing that says "this is the same experiment".

**`git_sha` returns `None`, never a placeholder (`config.py:99`).**
```python
return out.stdout.strip() or None
```
This repo is not currently a git repo, so every trace written today carries
`"git_sha": null`. That is the honest record. `git init` and a first commit turn it
into a real sha with no code change; a fabricated `"unknown"` string would have been
indistinguishable from a real value in a saved trace six weeks later. `git_is_dirty`
is recorded alongside, because a clean sha with uncommitted changes is a lie of a
different kind.

**Seeding stops short of full determinism (`config.py:120`).** `random`,
`PYTHONHASHSEED`, `numpy`, `torch` (+CUDA) are seeded; `torch.use_deterministic_algorithms`
and `cudnn.deterministic` are **not** set. Rationale in the docstring: cuDNN kernel
selection can still make two same-seed runs differ in the last decimal of an NLI
probability. Forcing determinism costs throughput, and if a *conclusion* ever depends
on that last decimal the conclusion is the problem. Revisit if a bug turns out to be
nondeterminism-shaped.

## 5. Data flow

```
configs/debug.yaml
   │ load_config
   ▼
dict{seed, pipeline, paths, components, harness, _config_path}
   │ apply_overrides(["components.retriever.name=bm25", "pipeline.k=10"])   ← CLI / sidebar
   ▼
resolved dict ──┬── registry.build_pipeline ──► Pipeline
                ├── config_hash ─────────────► Trace.config_hash, results/<run_id>/
                ├── set_global_seed(cfg["seed"])
                └── provenance() ────────────► results/<run_id>/provenance.json
```

## 6. How to verify it

```powershell
python -m pytest tests/test_config.py -v
```
Expected: **9 passed**.

| Test | Failure signature |
|---|---|
| `test_hash_ignores_underscore_keys_so_it_is_machine_independent` | the same config hashes differently on two machines; run dirs stop being comparable |
| `test_overrides_parse_values_as_yaml_scalars` | `k` arrives as `"10"`, the component compares it to an int, and everything downstream is subtly wrong |
| `test_overrides_do_not_mutate_the_input` | the sidebar's second re-run inherits the first's overrides — changes appear to "stick" at random |
| `test_git_sha_is_none_rather_than_fabricated_outside_a_repo` | traces carry a fake sha |

Manual check on the real config:

```powershell
python -c "from src.core.config import load_config, config_hash, provenance; c=load_config('configs/debug.yaml'); print(config_hash(c)); print(provenance(c))"
```
Expected: a 12-hex-digit hash, and `git_sha=None, git_dirty=None` until this
directory becomes a git repo.

## 7. Limitations and failure modes

* **`git_sha` is `None` for this whole build**, because `C:\Users\krish\Nullius` is
  not a git repo. Traces will record `null`. One `git init` + commit fixes it
  permanently; until then, provenance rests on `config_hash` alone. Logged in
  `OPEN_QUESTIONS.md`.
* **The hash covers the config, not the code, the data, or the model weights.** Two
  runs with the same hash and different `transformers` versions are different
  experiments. `requirements.lock.txt` (step 2) plus `git_sha` are what close that
  gap; the hash alone does not.
* **`apply_overrides` replaces leaves, it does not merge dicts.** `--set
  components.retriever='{name: bm25}'` replaces the whole spec including its params.
  Deliberate — a half-merged component spec is worse than an obvious replacement —
  but it will surprise you once.
* **No schema validation of the config.** A misspelled *top-level* key
  (`componets:`) is not caught here; it surfaces as `build_pipeline`'s "missing
  components" error, which names the right thing but not the typo.
* **`set_global_seed` seeds what is importable.** In a fresh venv without torch it
  silently seeds less. That is correct behaviour, but the trace does not record
  *which* RNGs were seeded.
* **`PYTHONHASHSEED` set at runtime does not affect the already-running
  interpreter's string hashing.** It only matters for subprocesses. Harmless here
  (nothing depends on string hash order), but it is not the guarantee its presence
  suggests.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Hydra** | Brings composition, overrides, and run dirs — 80% of what is needed — plus a config framework's semantics, a `conf/` layout convention, and an import-path instantiation model this repo explicitly rejected (see `core-registry.md` §8). 120 lines of stdlib is cheaper than the coupling. |
| **argparse-only, no config file** | Then "the resolved config" is a shell command someone has to have kept, and the sidebar has nothing to serialise. |
| **Hashing the config *file bytes*** | Whitespace and comment edits change the hash; a CLI override does not. Exactly backwards from what an experiment identifier needs. |
| **`hash()` / `md5`** | Python's `hash()` is salted per process, so it is not stable across runs at all. md5 works but invites a "why md5?" question with no upside over blake2b. |
| **Full determinism (`torch.use_deterministic_algorithms(True)`)** | Costs throughput, forces `CUBLAS_WORKSPACE_CONFIG` env juggling, and errors out on ops that have no deterministic kernel. Not worth it for an inspection harness; revisit if a nondeterminism-shaped bug appears. |
| **Storing provenance only in the run directory, not in the trace** | The failure-case files are meant to be readable standalone weeks later, possibly copied out of `results/`. Self-describing traces survive that; a sibling file does not. |
