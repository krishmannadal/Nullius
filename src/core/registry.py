"""Component registry: YAML names -> constructed components.

WHY THIS PATTERN (decorator registry with one namespace per interface), rather
than the two obvious alternatives:

  * ``_target_``-style import paths (Hydra / OmegaConf).  Rejected: the set of
    valid values is unbounded and unenumerable, so the Streamlit sidebar could not
    populate its dropdowns without a hand-maintained second list -- and a
    hand-maintained second list is a thing that goes stale.  It also executes an
    arbitrary import named in a config file, which is a foot-gun in a repo where
    configs get pasted around between machines.
  * ``entry_points`` / plugin discovery.  Rejected: solves distribution across
    packages, a problem this repo does not have, and costs import-time scanning
    plus a packaging step before a new aggregator can be tried.

What the decorator registry buys, concretely:
  1. ``available("aggregator")`` returns the exact list the UI dropdown needs, so
     "swap a component" really is a one-line config change and never a code change.
  2. Registration is namespaced per interface and type-checked at decoration time:
     registering a Retriever under "verifier" fails at import, not at run time.
  3. Unknown names fail with the list of valid names, and unknown *params* fail
     loudly instead of being silently ignored -- a YAML typo like ``top_k: 10``
     where the constructor says ``k`` is otherwise invisible and will quietly
     change your results.

Cost, stated honestly: components only exist in the registry once their module has
been imported.  Import-side-effect registration plus lazy discovery is a classic
source of "works in the test, missing in the app" bugs, so discovery here is
explicit (``load_builtins()`` imports a hard-coded list) rather than a filesystem
scan.  If you add a component package, you add one line to ``_BUILTIN_MODULES``.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, TypeVar

from src.core.interfaces import (
    Aggregator,
    ClaimExtractor,
    Component,
    Reranker,
    Retriever,
    Verifier,
)

#: interface kind -> base class.  The keys are exactly the keys allowed under
#: ``components:`` in a config file.
BASE_BY_KIND: dict[str, type[Component]] = {
    "extractor": ClaimExtractor,
    "retriever": Retriever,
    "reranker": Reranker,
    "verifier": Verifier,
    "aggregator": Aggregator,
}

KINDS: tuple[str, ...] = tuple(BASE_BY_KIND)

#: kind -> {name -> class}
_REGISTRY: dict[str, dict[str, type[Component]]] = {kind: {} for kind in KINDS}

#: Modules imported for their registration side effects.  Add a line here when you
#: add a component module; nothing scans the filesystem.
_BUILTIN_MODULES: tuple[str, ...] = (
    "src.components.extractors",
    "src.components.retrievers",
    "src.components.rerankers",
    "src.components.verifiers",
    "src.components.aggregators",
)

_builtins_loaded = False

C = TypeVar("C", bound=Component)


class RegistryError(KeyError):
    """Bad registration or bad lookup.  Always names the valid alternatives."""


def register(kind: str, name: str) -> Callable[[type[C]], type[C]]:
    """Class decorator: make ``cls`` constructible as ``kind`` named ``name``.

    Example::

        @register("retriever", "bm25")
        class BM25Retriever(Retriever): ...
    """
    if kind not in _REGISTRY:
        raise RegistryError(f"unknown component kind {kind!r}; valid kinds: {list(KINDS)}")

    def _decorate(cls: type[C]) -> type[C]:
        base = BASE_BY_KIND[kind]
        if not (inspect.isclass(cls) and issubclass(cls, base)):
            raise RegistryError(
                f"{cls!r} registered as {kind!r} but does not subclass {base.__name__}"
            )
        existing = _REGISTRY[kind].get(name)
        # Re-registering the identical class is fine (module reloaded under pytest);
        # re-registering a different class under a taken name is a silent-swap bug.
        if existing is not None and existing is not cls:
            raise RegistryError(
                f"{kind}:{name!r} already registered to {existing.__module__}.{existing.__qualname__}"
            )
        cls.registry_name = name
        _REGISTRY[kind][name] = cls
        return cls

    return _decorate


def load_builtins() -> None:
    """Import every module that registers a builtin component.  Idempotent."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    for module in _BUILTIN_MODULES:
        importlib.import_module(module)
    _builtins_loaded = True


def available(kind: str) -> list[str]:
    """Registered names for a kind, sorted.  This is what the UI dropdowns read."""
    if kind not in _REGISTRY:
        raise RegistryError(f"unknown component kind {kind!r}; valid kinds: {list(KINDS)}")
    load_builtins()
    return sorted(_REGISTRY[kind])


def available_all() -> dict[str, list[str]]:
    return {kind: available(kind) for kind in KINDS}


def get(kind: str, name: str) -> type[Component]:
    load_builtins()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise RegistryError(
            f"no {kind} named {name!r}; registered: {available(kind)}"
        ) from None


def normalise_spec(spec: Any) -> dict[str, Any]:
    """Accept ``"bm25"`` or ``{name: bm25, params: {...}}``; emit the dict form."""
    if isinstance(spec, str):
        return {"name": spec, "params": {}}
    if isinstance(spec, Mapping):
        if "name" not in spec:
            raise RegistryError(f"component spec {dict(spec)!r} has no 'name' key")
        params = spec.get("params") or {}
        if not isinstance(params, Mapping):
            raise RegistryError(f"'params' must be a mapping, got {type(params).__name__}")
        return {"name": str(spec["name"]), "params": dict(params)}
    raise RegistryError(f"component spec must be a string or mapping, got {type(spec).__name__}")


def _validate_params(cls: type[Component], params: Mapping[str, Any], kind: str, name: str) -> None:
    """Reject YAML keys the constructor does not accept.

    A mistyped key that is silently ignored changes results without changing
    anything visible, which is the exact failure mode this project cannot afford.
    Classes taking **kwargs opt out (nothing to check against).
    """
    sig = inspect.signature(cls.__init__)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    accepted = {n for n in sig.parameters if n != "self"}
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise RegistryError(
            f"{kind}:{name} got unknown params {unknown}; accepted: {sorted(accepted)}"
        )
    required = {
        n
        for n, p in sig.parameters.items()
        if n != "self"
        and p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    missing = sorted(required - set(params))
    if missing:
        raise RegistryError(f"{kind}:{name} missing required params {missing}")


def build(kind: str, spec: Any) -> Component:
    """Construct one component from a config spec."""
    norm = normalise_spec(spec)
    name, params = norm["name"], norm["params"]
    cls = get(kind, name)
    _validate_params(cls, params, kind, name)
    obj = cls(**params)  # type: ignore[call-arg]
    # Record what it was actually built with, for Trace.resolved_config.
    obj.resolved_params = dict(params)
    return obj


@dataclass(frozen=True, slots=True)
class Pipeline:
    """The five constructed components plus the k they run at."""

    extractor: ClaimExtractor
    retriever: Retriever
    reranker: Reranker
    verifier: Verifier
    aggregator: Aggregator
    k: int

    def describe(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "extractor": self.extractor.describe(),
            "retriever": self.retriever.describe(),
            "reranker": self.reranker.describe(),
            "verifier": self.verifier.describe(),
            "aggregator": self.aggregator.describe(),
        }


def build_pipeline(cfg: Mapping[str, Any]) -> Pipeline:
    """Build all five components from a resolved config dict.

    Expects::

        pipeline:
          k: 5
        components:
          extractor:  spacy_sentence
          retriever:  {name: hybrid, params: {rrf_k: 60}}
          reranker:   noop
          verifier:   nli_deberta
          aggregator: threshold_abstain

    A missing ``reranker`` defaults to ``noop`` rather than to ``None``: keeping the
    no-op on the same code path means the timing breakdown and the trace have the
    same shape whether or not reranking is on, so "reranker off" is a measurement,
    not a different program.
    """
    load_builtins()
    components = dict(cfg.get("components") or {})
    unknown_kinds = sorted(set(components) - set(KINDS))
    if unknown_kinds:
        raise RegistryError(f"config has unknown component kinds {unknown_kinds}; valid: {list(KINDS)}")
    components.setdefault("reranker", "noop")

    missing = [kind for kind in KINDS if kind not in components]
    if missing:
        raise RegistryError(f"config is missing components: {missing}")

    k = int((cfg.get("pipeline") or {}).get("k", 5))
    if k < 1:
        raise ValueError(f"pipeline.k must be >= 1, got {k}")

    return Pipeline(
        extractor=build("extractor", components["extractor"]),      # type: ignore[arg-type]
        retriever=build("retriever", components["retriever"]),      # type: ignore[arg-type]
        reranker=build("reranker", components["reranker"]),         # type: ignore[arg-type]
        verifier=build("verifier", components["verifier"]),         # type: ignore[arg-type]
        aggregator=build("aggregator", components["aggregator"]),   # type: ignore[arg-type]
        k=k,
    )


def _reset_for_tests(kind: Optional[str] = None) -> None:
    """Clear registrations.  Tests only -- never call this from application code."""
    global _builtins_loaded
    for key in ([kind] if kind else list(KINDS)):
        _REGISTRY[key].clear()
    _builtins_loaded = False


__all__ = [
    "KINDS",
    "BASE_BY_KIND",
    "RegistryError",
    "Pipeline",
    "register",
    "available",
    "available_all",
    "get",
    "build",
    "build_pipeline",
    "load_builtins",
    "normalise_spec",
]
