"""Download the 5 benchmark datasets, split, LOCK the held-out set, write manifest.

Run once (idempotent):

    python scripts/prepare_datasets.py

For each dataset in the registry this:
  1. Fetches it from OpenML at a pinned version (sklearn's fetch_openml).
  2. Standardises the target column to ``target``.
  3. Makes a train / holdout split (stratified for classification), seed-locked.
  4. Writes ``data/<key>/train.parquet`` and ``data/<key>/holdout.parquet``
     (both git-ignored — reproduced from this script, not committed).
  5. Records sizes, target info, and a SHA-256 of each split in
     ``results/dataset_manifest.json`` (committed — the reproducibility contract).

The holdout SHA-256 is the anchor for the no-peeking honesty guard. Because of
that, an existing manifest is a COMMITMENT, not a cache: when one is present
this script regenerates the data files and VERIFIES their frame seals against
it, leaving the committed manifest untouched — overwriting it would replace the
Day-1 commitment with a self-generated one and turn every downstream seal check
circular (it would also drop the Day-27 ``*_file_sha256`` byte seals, which
only ``scripts/add_file_seals.py`` writes). ``--force`` re-seals from scratch,
for first-time setup on a machine with no committed manifest to honour or a
deliberate, disclosed re-lock.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/prepare_datasets.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from crewml.config import HOLDOUT_FRACTION, SEED
from crewml.datasets import (
    MANIFEST_PATH,
    REGISTRY,
    TARGET_COLUMN,
    DatasetSpec,
    dataset_dir,
    holdout_path,
    sha256_of_frame,
    spec_asdict,
    train_path,
)


def _fetch(spec: DatasetSpec) -> pd.DataFrame:
    """Fetch a dataset and return a single frame with a ``target`` column."""
    bunch = fetch_openml(
        name=spec.openml_name,
        version=spec.version,
        as_frame=True,
        parser="auto",
    )
    X = bunch.data.copy()
    y = bunch.target.copy()
    if TARGET_COLUMN in X.columns:
        X = X.rename(columns={TARGET_COLUMN: f"{TARGET_COLUMN}_feature"})
    frame = X.copy()
    frame[TARGET_COLUMN] = y.values
    # Drop rows with a missing target — never impute the label.
    frame = frame.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return frame


def _target_summary(frame: pd.DataFrame, spec: DatasetSpec) -> dict:
    y = frame[TARGET_COLUMN]
    if spec.task == "classification":
        counts = y.value_counts().to_dict()
        return {
            "n_classes": int(y.nunique()),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "positive_class": str(sorted(counts, key=counts.get)[0])
            if spec.subtype == "binary" else None,  # rarer class = positive for AUC
        }
    return {
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
    }


def prepare_one(spec: DatasetSpec) -> dict:
    frame = _fetch(spec)
    stratify = frame[TARGET_COLUMN] if spec.task == "classification" else None
    train_df, holdout_df = train_test_split(
        frame,
        test_size=HOLDOUT_FRACTION,
        random_state=SEED,
        stratify=stratify,
    )
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    dataset_dir(spec.key).mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path(spec.key), index=False)
    holdout_df.to_parquet(holdout_path(spec.key), index=False)

    entry = {
        **spec_asdict(spec),
        "n_rows_total": int(len(frame)),
        "n_features": int(frame.shape[1] - 1),
        "n_train": int(len(train_df)),
        "n_holdout": int(len(holdout_df)),
        "target_column": TARGET_COLUMN,
        "target": _target_summary(frame, spec),
        "train_sha256": sha256_of_frame(train_df),
        "holdout_sha256": sha256_of_frame(holdout_df),
    }
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", action="store_true",
        help="overwrite an existing committed manifest instead of verifying "
             "against it (destroys the standing seal commitment — only for a "
             "deliberate re-lock)",
    )
    args = ap.parse_args(argv)

    datasets: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for key, spec in REGISTRY.items():
        print(f"[prepare] {key} (OpenML {spec.openml_name} v{spec.version}) ...", flush=True)
        try:
            entry = prepare_one(spec)
            datasets[key] = entry
            print(
                f"          ok — {entry['n_rows_total']} rows, "
                f"{entry['n_features']} feats, "
                f"train={entry['n_train']} holdout={entry['n_holdout']}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — report and continue
            failures[key] = f"{type(e).__name__}: {e}"
            print(f"          FAILED — {failures[key]}", flush=True)

    if MANIFEST_PATH.exists() and not args.force:
        # Verify mode: the committed manifest is the commitment; the fresh
        # split must reproduce its frame seals. (Byte seals are not compared —
        # parquet bytes vary across pandas versions; frames must not.)
        committed = json.loads(MANIFEST_PATH.read_text())["datasets"]
        mismatches: list[str] = []
        for key, entry in datasets.items():
            ref = committed.get(key)
            if ref is None:
                mismatches.append(f"{key}: not in the committed manifest")
                continue
            for seal in ("train_sha256", "holdout_sha256"):
                if entry[seal] != ref[seal]:
                    mismatches.append(
                        f"{key}.{seal}: regenerated {entry[seal][:12]}… != "
                        f"committed {ref[seal][:12]}…"
                    )
        if mismatches or failures:
            print("\n[prepare] SEAL VERIFICATION FAILED — the committed "
                  "manifest was left untouched:")
            for m in mismatches:
                print(f"          {m}")
            if failures:
                print(f"          fetch failures: {list(failures)}")
            return 1
        print(f"\n[prepare] seals verified: {len(datasets)}/{len(committed)} "
              f"datasets reproduce the committed manifest exactly.")
        print("[prepare] manifest left untouched (use --force to re-seal).")
        return 0

    manifest = {
        "seed": SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "target_column": TARGET_COLUMN,
        "n_datasets": len(datasets),
        "datasets": datasets,
        "failures": failures,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\n[prepare] wrote manifest -> {MANIFEST_PATH}")
    print(f"[prepare] {len(datasets)}/{len(REGISTRY)} datasets prepared.")
    print("[prepare] NOTE: a freshly-written manifest carries frame seals only; "
          "run scripts/add_file_seals.py to add the byte seals.")
    if failures:
        print(f"[prepare] failures: {list(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
