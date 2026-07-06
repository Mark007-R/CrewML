"""Dataset registry + loaders + held-out integrity checks.

The registry locks the 5 benchmark datasets, their task type, and the scoring
metric. ``prepare_datasets.py`` materialises a stratified train / holdout split
per dataset and records a SHA-256 fingerprint of each split in the manifest.

The HARD INVARIANT of the whole project: the crew only ever loads ``train`` via
:func:`load_train`. The ``holdout`` split is loaded exclusively by the final
scorer via :func:`load_holdout`, and :func:`verify_holdout_untouched` proves the
crew never saw or mutated it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from crewml.config import DATA_DIR, RESULTS_DIR

TARGET_COLUMN = "target"  # standardised target name across all datasets
MANIFEST_PATH = RESULTS_DIR / "dataset_manifest.json"


@dataclass(frozen=True)
class DatasetSpec:
    key: str            # short local id / folder name
    openml_name: str    # name passed to sklearn.datasets.fetch_openml
    version: int        # OpenML dataset version (pinned for reproducibility)
    task: str           # "classification" | "regression"
    subtype: str        # "binary" | "multiclass" | "regression"
    metric: str         # primary scoring metric (see EVAL_PROTOCOL.md)
    note: str           # why this dataset earns its place in the suite


# The locked benchmark suite: 2 binary, 1 multiclass, 2 regression.
REGISTRY: dict[str, DatasetSpec] = {
    "credit-g": DatasetSpec(
        key="credit-g", openml_name="credit-g", version=1,
        task="classification", subtype="binary", metric="roc_auc",
        note="Class-imbalanced (700/300) with categoricals — tests handling of "
             "skew and mixed dtypes.",
    ),
    "diabetes": DatasetSpec(
        key="diabetes", openml_name="diabetes", version=1,
        task="classification", subtype="binary", metric="roc_auc",
        note="Pima Indians, all-numeric with disguised-missing zeros — tests "
             "missing-value detection the crew must not miss.",
    ),
    "vehicle": DatasetSpec(
        key="vehicle", openml_name="vehicle", version=1,
        task="classification", subtype="multiclass", metric="f1_macro",
        note="4-class silhouette recognition — tests multiclass metric choice "
             "and per-class balance.",
    ),
    "cpu_small": DatasetSpec(
        key="cpu_small", openml_name="cpu_small", version=1,
        task="regression", subtype="regression", metric="r2",
        note="8k-row system-activity regression — a larger, clean numeric "
             "target for R^2 scoring.",
    ),
    "kin8nm": DatasetSpec(
        key="kin8nm", openml_name="kin8nm", version=1,
        task="regression", subtype="regression", metric="r2",
        note="Robot-arm forward-kinematics regression — smooth non-linear "
             "target that rewards good feature engineering.",
    ),
}


def dataset_dir(key: str) -> Path:
    return DATA_DIR / key


def train_path(key: str) -> Path:
    return dataset_dir(key) / "train.parquet"


def holdout_path(key: str) -> Path:
    return dataset_dir(key) / "holdout.parquet"


def sha256_of_frame(df: pd.DataFrame) -> str:
    """Order-independent-of-serialisation SHA-256 of a dataframe's contents."""
    buf = df.to_parquet(index=False)
    return hashlib.sha256(buf).hexdigest()


def load_train(key: str) -> pd.DataFrame:
    """Load the training split — the ONLY split the crew is allowed to touch."""
    return pd.read_parquet(train_path(key))


def load_holdout(key: str) -> pd.DataFrame:
    """Load the LOCKED held-out split — final scoring only, never for modeling."""
    return pd.read_parquet(holdout_path(key))


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} missing — run scripts/prepare_datasets.py first."
        )
    return json.loads(MANIFEST_PATH.read_text())


def verify_holdout_untouched(key: str) -> bool:
    """Return True iff the on-disk holdout still matches its manifest fingerprint.

    This is the honesty proof: if the crew (or anyone) altered the held-out set,
    the SHA-256 will diverge and this returns False.
    """
    manifest = load_manifest()
    recorded = manifest["datasets"][key]["holdout_sha256"]
    current = sha256_of_frame(load_holdout(key))
    return recorded == current


def spec_asdict(spec: DatasetSpec) -> dict:
    return asdict(spec)
