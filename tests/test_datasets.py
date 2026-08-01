"""Day 1 guards: the datasets are locked, split cleanly, and the holdout is sealed.

These tests are the foundation of the honesty story. If any fail, no downstream
crew/baseline result can be trusted, so they run first in CI.
"""
from __future__ import annotations

import pandas as pd
import pytest

from crewml.datasets import (
    BENCHMARK_KEYS,
    REGISTRY,
    TARGET_COLUMN,
    holdout_path,
    load_holdout,
    load_manifest,
    load_train,
    train_path,
    verify_holdout_untouched,
)

KEYS = list(BENCHMARK_KEYS)  # benchmark-scoped, immune to restored uploads


def _require_prepared(key: str) -> None:
    if not (train_path(key).exists() and holdout_path(key).exists()):
        pytest.skip(f"{key} not materialised — run scripts/prepare_datasets.py")


def test_registry_has_expected_mix():
    subtypes = sorted(REGISTRY[k].subtype for k in BENCHMARK_KEYS)
    assert subtypes == ["binary", "binary", "multiclass", "regression", "regression"]


def test_manifest_present_and_complete():
    manifest = load_manifest()
    assert manifest["seed"] == 42
    assert set(manifest["datasets"]) == set(BENCHMARK_KEYS)
    assert manifest["failures"] == {}


@pytest.mark.parametrize("key", KEYS)
def test_target_column_standardised(key):
    _require_prepared(key)
    for frame in (load_train(key), load_holdout(key)):
        assert TARGET_COLUMN in frame.columns
        assert frame[TARGET_COLUMN].isna().sum() == 0


@pytest.mark.parametrize("key", KEYS)
def test_train_holdout_disjoint(key):
    """No row leaks from train into the held-out set."""
    _require_prepared(key)
    train, holdout = load_train(key), load_holdout(key)
    merged = train.merge(holdout, how="inner")
    assert len(merged) == 0, f"{len(merged)} rows shared between train and holdout"


@pytest.mark.parametrize("key", KEYS)
def test_holdout_fraction_reasonable(key):
    _require_prepared(key)
    n_train, n_hold = len(load_train(key)), len(load_holdout(key))
    frac = n_hold / (n_train + n_hold)
    assert 0.15 <= frac <= 0.25


@pytest.mark.parametrize("key", KEYS)
def test_holdout_untouched(key):
    """The no-peeking anchor: on-disk holdout matches its manifest fingerprint."""
    _require_prepared(key)
    assert verify_holdout_untouched(key) is True


def test_tampering_is_detected():
    """Sanity-check the guard itself: a mutated holdout must fail verification."""
    key = KEYS[0]
    _require_prepared(key)
    manifest = load_manifest()
    recorded = manifest["datasets"][key]["holdout_sha256"]
    from crewml.datasets import sha256_of_frame

    tampered = load_holdout(key).copy()
    tampered.iloc[0, 0] = tampered.iloc[0, 0]  # no-op keeps hash equal
    assert sha256_of_frame(tampered) == recorded
    # Now a real mutation must diverge.
    tampered = pd.concat([tampered, tampered.iloc[[0]]], ignore_index=True)
    assert sha256_of_frame(tampered) != recorded
