# `src/core/registry.py` — YAML names → components

## 1. Problem

The entire justification for the ABCs is that swapping a component is a one-line
config change and never a code change. That requires three things at once:

1. A **name → class** mapping the config can address.
2. An **enumerable** set of valid names, because the Streamlit sidebar has to
   populate five dropdowns and a hand-maintained duplicate list will go stale the
   first time you add an aggregator at 1 a.m.
3. **Loud failure on config mistakes.** `top_k: 10` where the constructor says `k`
   is otherwise silently ignored: the run succeeds, the number changes, and nothing
   anywhere records that your override did nothing. That is the exact failure mode
   that turns a week of results into noise.

## 2. Formal I/O

```
register(kind, name)          : (str, str) → (type[C] → type[C])     # class decorator
available(kind)               : str → list[str]                       # sorted, what the UI reads
available_all()               : () → dict[str, list[str]]
get(kind, name)               : (str, str) → type[Component]
normalise_spec(spec)          : str | Mapping → {"name": str, "params": dict}
build(kind, spec)             : (str, spec) → Component               # params validated
build_pipeline(cfg)           : Mapping → Pipeline
```

`kind ∈ {"extractor", "retriever", "reranker", "verifier", "aggregator"}`, which are
exactly the allowed keys under `components:` in a config.

Config shape consumed by `build_pipeline`:

```yaml
pipeline:   {k: 5}
components:
  extractor:  spacy_sentence                      # shorthand: name only
  retriever:  {name: hybrid, params: {rrf_k: 60}} # full form
  reranker:   noop                                # defaulted to "noop" if absent
  verifier:   nli_deberta
  aggregator: {name: threshold_abstain, params: {support_floor: 0.5}}
```

`Pipeline` is a frozen dataclass of the five instances plus `k`, and
`Pipeline.describe()` returns the JSON-shaped provenance block that goes into
`Trace.resolved_config`.

## 3. Algorithm

```
register(kind, name)(cls):
    assert kind ∈ KINDS
    assert issubclass(cls, BASE_BY_KIND[kind])        # wrong-slot registration fails at import
    if name taken by a DIFFERENT class: raise         # same class again = no-op (pytest reload)
    cls.registry_name ← name                          # single source of truth for trace names
    _REGISTRY[kind][name] ← cls

build(kind, spec):
    name, params ← normalise_spec(spec)
    cls ← get(kind, name)                             # KeyError lists valid names
    validate_params(cls, params)                      # unknown → raise; missing required → raise
    obj ← cls(**params)
    obj.resolved_params ← params                      # so the trace records the actual config
    return obj
```

## 4. Code walkthrough — the lines that matter

**Wrong-slot registration fails at import (`registry.py:97`).**
```python
if not (inspect.isclass(cls) and issubclass(cls, base)):
    raise RegistryError(f"{cls!r} registered as {kind!r} but does not subclass {base.__name__}")
```
Registering a `Retriever` under `"verifier"` raises when the module is imported, not
when a config eventually names it. This is why `Component` subclasses ABCs rather
than being duck-typed (see the Protocol row in `core-interfaces.md` §8).

**The single source of truth for names (`registry.py:108`).**
```python
cls.registry_name = name
```
The string you type in YAML *is* the string that appears in
`Evidence.retriever_name` and `ClaimVerdict.aggregator_name`. There is no second
place to keep in sync, so a trace can never disagree with the config that produced
it.

**Same-class re-registration is a no-op, different-class is an error
(`registry.py:104`).**
```python
if existing is not None and existing is not cls:
    raise RegistryError(f"{kind}:{name!r} already registered to ...")
```
Under pytest a module can be imported twice; failing on that would be noise.
Failing on a *different* class taking a live name is essential — otherwise the
config says `bm25` and you silently get someone else's `bm25`.

**Unknown-param rejection (`registry.py:168`).**
```python
sig = inspect.signature(cls.__init__)
if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
    return                                    # **kwargs classes opt out; nothing to check
accepted = {n for n in sig.parameters if n != "self"}
unknown  = sorted(set(params) - accepted)
```
The most valuable ten lines in the file. Turns a silent no-op typo into
`RegistryError: retriever:hybrid got unknown params ['top_k']; accepted: ['candidates_per_arm', 'rrf_k']`.
The `**kwargs` escape hatch exists because a component that legitimately forwards
kwargs (e.g. to a `SentenceTransformer`) has nothing to validate against; such a
component gives up this protection and its doc should say so.

**Reranker defaults to `noop`, never to `None` (`registry.py:247`).**
```python
components.setdefault("reranker", "noop")
```
Keeping the no-op on the same code path means the trace and the timing breakdown
have identical shape whether reranking is on or off. "Reranker off" then becomes a
*measurement* (a `noop` entry with its own latency and its own name in the trace)
rather than a different program with an `if reranker is not None` branch in it.

**Explicit discovery (`registry.py:66`).** `_BUILTIN_MODULES` is a hardcoded
tuple, imported by `load_builtins()`. No filesystem scan, no `pkgutil.walk_packages`.
Adding a component package costs one line here; the payoff is that the registry
contents are deterministic and greppable, so "works in the test, missing in the app"
cannot happen.

## 5. Data flow

```
configs/debug.yaml ──load_config──► dict ──apply_overrides──► resolved dict
                                                                │
                    ┌───────────────────────────────────────────┤
                    ▼                                           ▼
        registry.build_pipeline(cfg)                    config.config_hash(cfg)
                    │                                           │
                    ▼                                           ▼
              Pipeline (5 objects + k)  ──describe()──► Trace.resolved_config / config_hash

Streamlit sidebar ──► registry.available_all() ──► five dropdowns
Sidebar change    ──► apply_overrides([...]) ──► build_pipeline ──► re-run ──► re-render
```

## 6. How to verify it

```powershell
python -m pytest tests/test_registry.py -v
```
Expected: **14 passed**. The two that matter most:

| Test | Failure signature if it breaks |
|---|---|
| `test_unknown_param_is_rejected_not_ignored` | a YAML typo silently changes nothing; you spend a day comparing two runs that were the same run |
| `test_duplicate_name_with_a_different_class_is_rejected` | the config says `bm25`, you get a different `bm25`, and the trace records the name you expected |

Once components land (step 2), this becomes the live check:

```powershell
python -c "from src.core import registry; print(registry.available_all())"
```
Expected today: `{'extractor': [], 'retriever': [], 'reranker': [], 'verifier': [], 'aggregator': []}` —
correct, because `_BUILTIN_MODULES` is still empty. `configs/debug.yaml` therefore
fails to build with a `RegistryError` naming the empty list, which is the intended
behaviour for a config written ahead of its components.

## 7. Limitations and failure modes

* **Registration requires import.** A component in a module nobody imports does not
  exist. Mitigated by the explicit `_BUILTIN_MODULES` list, but the failure mode —
  "I added a class and the dropdown does not show it" — will happen at least once.
  The fix is always the same one line.
* **`_validate_params` inspects `__init__` only.** A component using `__new__`, a
  metaclass, or `functools.partial` wrappers will confuse it.
* **No type checking of param *values*.** `rrf_k: "sixty"` passes the registry and
  fails inside the component. Deliberate: doing it properly means a schema per
  component, which is the pydantic-shaped complexity this layer is avoiding.
  Components validate their own values in `__init__`.
* **Global mutable state.** `_REGISTRY` is module-level, so tests must save/restore
  it (`tests/test_registry.py::scratch_registry` does). Two pipelines with different
  registries cannot coexist in one process — not a use case here.
* **`build_pipeline` constructs everything eagerly**, so switching a dropdown in the
  UI reloads the model behind that dropdown. Step 5/6 will need an instance cache
  keyed on `(kind, name, params)`; noted, not built.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Hydra / OmegaConf `_target_` import paths** | The valid-value set is unbounded and unenumerable, so the sidebar dropdowns would need a hand-maintained parallel list. It also executes an arbitrary import named in a config file — a foot-gun for configs pasted between machines — and pulls in a config framework whose composition semantics are a second thing to learn. |
| **`importlib.metadata` entry points / plugin discovery** | Solves cross-package distribution, a problem this repo does not have, and requires a packaging step before a new aggregator can be tried. The whole point is that trying a new aggregator should take 60 seconds. |
| **`pkgutil.walk_packages` auto-discovery** | Import-order-dependent, imports everything (so a broken experimental file breaks the app), and makes "which classes are registered?" unanswerable without running it. The explicit list is greppable. |
| **A dict literal `{"bm25": BM25Retriever, ...}` in one module** | Needs a top-level import of every component, so the frontend would pull torch and faiss just to render a saved trace. Decorator registration keeps that import lazy. |
| **`__init_subclass__` auto-registration** | Registers by class name, so the config is coupled to Python identifiers; renaming a class silently invalidates every config and every saved trace that names it. |
| **Passing constructed objects around instead of a registry** | Works for a script, fails for the UI: the sidebar needs to construct a component from a string chosen at runtime. |
