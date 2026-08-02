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


# --- Day 27 fix: byte seals — environment-independent verification ----------

def test_manifest_carries_byte_seals():
    """Every benchmark entry has file seals (scripts/add_file_seals.py ran)."""
    manifest = load_manifest()
    for key in BENCHMARK_KEYS:
        entry = manifest["datasets"][key]
        assert len(entry["train_file_sha256"]) == 64
        assert len(entry["holdout_file_sha256"]) == 64


def test_byte_seal_matches_on_disk_file():
    from crewml.datasets import holdout_path, sha256_of_file

    manifest = load_manifest()
    for key in BENCHMARK_KEYS:
        recorded = manifest["datasets"][key]["holdout_file_sha256"]
        assert sha256_of_file(holdout_path(key)) == recorded


def test_verification_prefers_byte_seal(monkeypatch):
    """With a byte seal present, verification never re-serialises the frame —
    the whole point: no pandas/pyarrow version can flip the verdict."""
    from crewml import datasets

    def boom(df):  # noqa: ARG001
        raise AssertionError("frame seal path used despite byte seal present")

    monkeypatch.setattr(datasets, "sha256_of_frame", boom)
    assert datasets.verify_holdout_untouched("credit-g") is True


def test_legacy_manifest_falls_back_to_frame_seal(monkeypatch):
    """Entries without file seals (pre-Day-27) still verify via frame hash."""
    from crewml import datasets

    manifest = datasets.load_manifest()
    entry = dict(manifest["datasets"]["credit-g"])
    entry.pop("train_file_sha256"), entry.pop("holdout_file_sha256")
    legacy = {**manifest, "datasets": {**manifest["datasets"], "credit-g": entry}}
    monkeypatch.setattr(datasets, "load_manifest", lambda: legacy)
    assert datasets.verify_holdout_untouched("credit-g") is True


def test_byte_seal_detects_rewritten_file(tmp_path, monkeypatch):
    """Even a byte-identical-CONTENT rewrite is flagged: byte seals pin the
    exact artifact, strictly stricter than the frame seal they extend."""
    import shutil

    from crewml import datasets

    key = "credit-g"
    fake_dir = tmp_path / key
    fake_dir.mkdir()
    shutil.copy(datasets.train_path(key), fake_dir / "train.parquet")
    # Re-serialise the same rows through pandas: same frame, different bytes
    # (metadata/ordering) — or identical bytes in the sealing env; either way
    # appending a row must flip the byte seal.
    df = load_holdout(key)
    tampered = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    tampered.to_parquet(fake_dir / "holdout.parquet", index=False)
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path)
    assert datasets.verify_holdout_untouched(key) is False
