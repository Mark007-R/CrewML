"""Day 22: the no-peeking invariant, pinned as a regression suite.

The project's headline claim is that the crew never saw the holdout. Up to
Day 21 that claim rested on three layered guarantees, each tested where it
lives: the SHA-256 seal (test_datasets), static no-reference sweeps of the crew
package and executor (test_graph / test_executor), and the Day-19 sandbox's
read-deny root. This file adds the two pieces Day 22 closes out:

* **Runtime demonstration** — sandboxed code that *tries* to read the holdout
  by absolute path is refused at the moment of use, while its staged train
  input stays readable. The static sweeps prove our code doesn't name the
  loader; this proves generated code that goes around the loader still fails.
* **Suite-wide sweep** — one call seal-checks every manifest-locked dataset,
  and the new Day-22 modules are held to the same never-name-the-holdout
  standard as the crew package.
"""
from __future__ import annotations

import inspect

import pytest

from crewml import leakage
from crewml.datasets import holdout_path, train_path, verify_all_holdouts
from crewml.executor import run_code

KEY = "credit-g"


def _require_prepared(key: str) -> None:
    if not (train_path(key).exists() and holdout_path(key).exists()):
        pytest.skip(f"{key} not materialised — run scripts/prepare_datasets.py")


# --- Runtime no-peek: the sandbox refuses the read at the moment of use ------

def test_sandboxed_code_cannot_read_the_holdout_by_path():
    _require_prepared(KEY)
    script = (
        "import pandas as pd\n"
        "from crew_io import input_path\n"
        "df = pd.read_parquet(input_path('train.parquet'))\n"
        "print('staged read ok', len(df), flush=True)\n"
        f"pd.read_parquet(r'{holdout_path(KEY)}')\n"
        "print('HOLDOUT_READ_SUCCEEDED', flush=True)\n"
    )
    result = run_code(
        script,
        inputs={"train.parquet": train_path(KEY)},
        timeout_s=120,
        keep_workdir=False,
    )
    assert "staged read ok" in result.stdout          # the control: inputs work
    assert "HOLDOUT_READ_SUCCEEDED" not in result.stdout
    assert result.ok is False
    assert "denied root" in (result.error or "")      # refused by the guard, not luck


def test_sandboxed_open_of_the_holdout_is_also_refused():
    """Same invariant one layer down: a plain open(), no pandas involved."""
    _require_prepared(KEY)
    script = f"open(r'{holdout_path(KEY)}', 'rb').read(10)\n"
    result = run_code(script, timeout_s=60, keep_workdir=False)
    assert result.ok is False
    assert "denied root" in (result.error or "")


# --- Suite-wide seal sweep ----------------------------------------------------

def test_every_locked_holdout_is_sealed():
    _require_prepared(KEY)
    seals = verify_all_holdouts()
    assert len(seals) == 5
    assert all(seals.values()), f"broken seal(s): {[k for k, v in seals.items() if not v]}"


def test_seal_sweep_ignores_probe_registry_entries():
    """Probe datasets (train-only throwaways) must not break — or appear in —
    the sweep: it iterates the manifest, not the live REGISTRY."""
    import numpy as np

    from crewml.datasets import load_train
    from crewml.failure_taxonomy import make_leak_frame, probe_dataset

    _require_prepared(KEY)
    from crewml.datasets import REGISTRY

    spec = REGISTRY[KEY]
    frame, _ = make_leak_frame(
        load_train(KEY), task=spec.task, kind="subtle",
        rng=np.random.default_rng(0),
    )
    with probe_dataset(KEY, "probe_test_seal_sweep", frame):
        seals = verify_all_holdouts()
    assert "probe_test_seal_sweep" not in seals
    assert len(seals) == 5


# --- Static: the Day-22 modules obey the same standard ------------------------

def test_leakage_screen_never_references_the_holdout():
    # Same standard the crew package is held to: never name the loader.
    src = inspect.getsource(leakage)
    assert "load_holdout" not in src
    assert "holdout_path" not in src
