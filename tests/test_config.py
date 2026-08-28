"""Contract tests for src/core/config.py — provenance must be reproducible."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core import config as cfgmod

YAML = """
seed: 13
pipeline:
  k: 5
components:
  retriever: {name: bm25, params: {k1: 1.2}}
"""


@pytest.fixture()
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "debug.yaml"
    p.write_text(YAML, encoding="utf-8")
    return p


def test_load_config_records_its_own_path(cfg_file: Path):
    cfg = cfgmod.load_config(cfg_file)
    assert cfg["seed"] == 13
    assert Path(cfg["_config_path"]) == cfg_file.resolve()


def test_hash_ignores_underscore_keys_so_it_is_machine_independent(cfg_file: Path):
    cfg = cfgmod.load_config(cfg_file)
    other = dict(cfg, _config_path="/somewhere/else/debug.yaml")
    assert cfgmod.config_hash(cfg) == cfgmod.config_hash(other)


def test_hash_is_insensitive_to_key_order():
    a = {"pipeline": {"k": 5}, "seed": 1}
    b = {"seed": 1, "pipeline": {"k": 5}}
    assert cfgmod.config_hash(a) == cfgmod.config_hash(b)


def test_hash_changes_when_any_value_changes():
    a = {"pipeline": {"k": 5}}
    b = {"pipeline": {"k": 6}}
    assert cfgmod.config_hash(a) != cfgmod.config_hash(b)


def test_overrides_parse_values_as_yaml_scalars(cfg_file: Path):
    cfg = cfgmod.load_config(cfg_file)
    out = cfgmod.apply_overrides(cfg, ["pipeline.k=10", "components.retriever.name=dense"])
    assert out["pipeline"]["k"] == 10  # int, not "10"
    assert out["components"]["retriever"]["name"] == "dense"
    # ...and the same value typed in YAML would hash identically:
    assert cfgmod.config_hash(out) == cfgmod.config_hash(
        {**{k: v for k, v in cfg.items() if not k.startswith("_")},
         "pipeline": {"k": 10},
         "components": {"retriever": {"name": "dense", "params": {"k1": 1.2}}}}
    )


def test_overrides_do_not_mutate_the_input(cfg_file: Path):
    cfg = cfgmod.load_config(cfg_file)
    cfgmod.apply_overrides(cfg, ["pipeline.k=99"])
    assert cfg["pipeline"]["k"] == 5


def test_malformed_override_is_rejected(cfg_file: Path):
    with pytest.raises(ValueError, match="must look like"):
        cfgmod.apply_overrides(cfgmod.load_config(cfg_file), ["pipeline.k"])


def test_git_sha_is_none_rather_than_fabricated_outside_a_repo(tmp_path: Path):
    assert cfgmod.git_sha(tmp_path) is None


def test_set_global_seed_makes_python_rng_reproducible():
    import random

    cfgmod.set_global_seed(7)
    first = [random.random() for _ in range(3)]
    cfgmod.set_global_seed(7)
    assert [random.random() for _ in range(3)] == first


def test_configure_console_is_safe_on_captured_streams():
    """pytest replaces sys.stdout with a non-reconfigurable capture object; the helper
    must degrade silently rather than raise."""
    cfgmod.configure_console()  # must not raise
    print("smoke: ́ 中文 —")  # combining acute, CJK, em dash
