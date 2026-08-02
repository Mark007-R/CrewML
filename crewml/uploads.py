"""Uploaded-CSV ingestion — Day 26: user data enters under the same honesty rules.

The Day-1 benchmark suite gets its honesty guarantees from
``scripts/prepare_datasets.py``: a known OpenML label becomes the ``target``
column, the split is seed-locked, and the holdout is SHA-256 sealed into a
manifest before any agent runs. An uploaded CSV arrives with none of that, so
this module rebuilds every piece of it at ingestion time:

* **The target column is CHOSEN, never guessed.** The caller must name it; there
  is no fallback to "the last column" or "a column called y". Silently modeling
  the wrong column would produce a confident, fully-reported, meaningless model
  — so a missing or ambiguous choice is an error, not a heuristic.
* **Task / subtype / metric are DERIVED, then shown.** :func:`derive_target`
  applies the EVAL_PROTOCOL rules (binary→roc_auc, multiclass→f1_macro,
  regression→r2) from the chosen column's dtype + cardinality, and the derivation
  (with the evidence: dtype, n_unique, class counts) is returned so a wrong pick
  is visible *before* the run starts.
* **The holdout is sealed at ingestion.** The split happens here — server-side,
  seed-locked, stratified for classification — and both split SHA-256s go into a
  per-upload manifest (git-ignored, beside the data). ``manifest.dataset_seals``
  and ``datasets.verify_holdout_untouched`` read uploads from that manifest, so a
  crew run on user data carries the same seal evidence as a benchmark run.
* **Structural no-peeking is unchanged.** An upload registers as an ordinary
  ``DatasetSpec``; ``CrewState`` still carries only ``dataset_key``, the crew
  still loads ``train`` via ``load_train``, and nothing hands it a holdout path.

Upload keys are content-addressed (``upload-<sha12>`` over the CSV bytes + the
chosen target), so re-ingesting the same file with the same choice is idempotent
and re-uses the sealed split instead of minting a competing one.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split

from crewml.config import DATA_DIR, HOLDOUT_FRACTION, SEED
from crewml.datasets import (
    TARGET_COLUMN,
    DatasetSpec,
    REGISTRY,
    dataset_dir,
    holdout_path,
    sha256_of_file,
    sha256_of_frame,
    spec_asdict,
    train_path,
)

UPLOAD_PREFIX = "upload-"
UPLOAD_MANIFEST_NAME = "upload_manifest.json"

# Derivation thresholds (EVAL_PROTOCOL §2 applied to unlabeled data). An
# integer-coded column with few distinct values reads as class labels; many
# distinct values read as a quantity. 20 is generous for real label sets while
# far below anything plausibly continuous.
MAX_CLASSES = 50            # beyond this a "classification" pick is almost
                            # certainly an ID / free-text column — reject loudly
INT_AS_CLASS_MAX_UNIQUE = 20
ID_LIKE_UNIQUE_RATIO = 0.99  # integer target ~1:1 with rows → warn: looks like an id

MIN_ROWS = 30  # below this a stratified 80/20 split stops being meaningful


class UploadError(ValueError):
    """A rejected upload — the message is safe to surface to the caller."""


@dataclass
class TargetDerivation:
    """What we concluded about the chosen target column, with the evidence."""

    task: str        # "classification" | "regression"
    subtype: str     # "binary" | "multiclass" | "regression"
    metric: str      # "roc_auc" | "f1_macro" | "r2"
    dtype: str
    n_unique: int
    rule: str        # the human-readable reason for the derivation
    warnings: list[str] = field(default_factory=list)
    target_summary: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return {
            "task": self.task, "subtype": self.subtype, "metric": self.metric,
            "dtype": self.dtype, "n_unique": self.n_unique, "rule": self.rule,
            "warnings": self.warnings, "target_summary": self.target_summary,
        }


def is_upload_key(key: str) -> bool:
    return key.startswith(UPLOAD_PREFIX)


def upload_manifest_path(key: str) -> Path:
    return dataset_dir(key) / UPLOAD_MANIFEST_NAME


def load_upload_manifest(key: str) -> dict[str, Any]:
    path = upload_manifest_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"no upload manifest for {key!r} — was this dataset ingested via "
            f"crewml.uploads.ingest_csv?"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def derive_target(y: pd.Series) -> TargetDerivation:
    """Derive (task, subtype, metric) from the CHOSEN column — never pick the column.

    Rules, in order (documented because the UI shows the winning one):

    1. n_unique < 2 → reject: a constant column cannot be a target.
    2. bool / non-numeric dtype → classification.
    3. integer dtype with n_unique <= 20 → classification (integer class codes).
    4. any other numeric → regression (metric r2).
    5. classification with n_unique == 2 → binary (roc_auc); 3..50 → multiclass
       (f1_macro); > 50 → reject: that many "classes" means the pick is almost
       certainly an identifier or free text, and a run would be meaningless.
    """
    y = y.dropna()
    n_unique = int(y.nunique())
    dtype = str(y.dtype)
    warnings: list[str] = []

    if n_unique < 2:
        raise UploadError(
            f"target column has {n_unique} distinct value(s) — a constant "
            f"column cannot be predicted. Pick a different column."
        )

    numeric = is_numeric_dtype(y) and not is_bool_dtype(y)
    if numeric and (not is_integer_dtype(y) or n_unique > INT_AS_CLASS_MAX_UNIQUE):
        rule = (f"numeric dtype ({dtype}) with {n_unique} distinct values "
                f"→ regression, scored by R²")
        if is_integer_dtype(y) and n_unique >= ID_LIKE_UNIQUE_RATIO * len(y):
            warnings.append(
                f"target is integer with {n_unique} distinct values over "
                f"{len(y)} rows — this looks like a row identifier, not a "
                f"quantity. Double-check the column choice."
            )
        summary = {
            "target_min": float(y.min()), "target_max": float(y.max()),
            "target_mean": float(y.mean()),
        }
        return TargetDerivation(
            task="regression", subtype="regression", metric="r2",
            dtype=dtype, n_unique=n_unique, rule=rule,
            warnings=warnings, target_summary=summary,
        )

    # classification: non-numeric, boolean, or low-cardinality integer codes
    if n_unique > MAX_CLASSES:
        raise UploadError(
            f"target column has {n_unique} distinct non-numeric values — more "
            f"than {MAX_CLASSES} classes almost always means an identifier or "
            f"free-text column was picked. Choose the actual label column."
        )
    counts = y.astype(str).value_counts()
    summary = {
        "n_classes": n_unique,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
    }
    if n_unique == 2:
        # rarer class = positive, matching prepare_datasets.py / EVAL_PROTOCOL
        summary["positive_class"] = str(counts.index[-1])
        rule = (f"{dtype} dtype with exactly 2 distinct values → binary "
                f"classification, scored by ROC AUC on the rarer class")
        subtype, metric = "binary", "roc_auc"
    else:
        rule = (f"{dtype} dtype with {n_unique} distinct values → multiclass "
                f"classification, scored by macro-F1")
        subtype, metric = "multiclass", "f1_macro"
    minority = int(counts.iloc[-1])
    if minority < 5:
        warnings.append(
            f"rarest class has only {minority} row(s) — stratified CV will "
            f"struggle; consider whether this column is really the label."
        )
    return TargetDerivation(
        task="classification", subtype=subtype, metric=metric,
        dtype=dtype, n_unique=n_unique, rule=rule,
        warnings=warnings, target_summary=summary,
    )


def _upload_key(csv_bytes: bytes, target_column: str) -> str:
    digest = hashlib.sha256()
    digest.update(csv_bytes)
    digest.update(b"\x00target=")
    digest.update(target_column.encode("utf-8"))
    return UPLOAD_PREFIX + digest.hexdigest()[:12]


def _safe_name(filename: Optional[str]) -> str:
    stem = Path(filename or "upload").stem or "upload"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:64]


def ingest_csv(csv_bytes: bytes, *, target_column: str,
               filename: Optional[str] = None) -> dict[str, Any]:
    """Ingest an uploaded CSV: validate, derive, split, SEAL, register.

    Returns the per-upload manifest (also persisted beside the splits). Raises
    :class:`UploadError` for anything the uploader can fix — bad CSV, unknown
    target column, undecidable target — with a message safe to show them.
    """
    if not csv_bytes or not csv_bytes.strip():
        raise UploadError("the uploaded file is empty.")
    if not target_column:
        raise UploadError(
            "no target column was chosen. The target is always chosen "
            "explicitly — CrewML never guesses which column to predict."
        )
    try:
        frame = pd.read_csv(BytesIO(csv_bytes))
    except Exception as exc:
        raise UploadError(f"could not parse the file as CSV: {exc}") from None
    if frame.empty or frame.shape[1] == 0:
        raise UploadError("the CSV parsed to an empty table.")
    if target_column not in frame.columns:
        raise UploadError(
            f"column {target_column!r} is not in the CSV. Available columns: "
            f"{list(frame.columns)}"
        )

    # Same normalisation prepare_datasets.py applies to OpenML data: the chosen
    # column BECOMES `target`; a pre-existing unrelated `target` column steps
    # aside as a feature rather than colliding.
    if TARGET_COLUMN in frame.columns and target_column != TARGET_COLUMN:
        frame = frame.rename(columns={TARGET_COLUMN: f"{TARGET_COLUMN}_feature"})
    frame = frame.rename(columns={target_column: TARGET_COLUMN})

    n_before = len(frame)
    frame = frame.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    n_dropped = n_before - len(frame)
    if len(frame) < MIN_ROWS:
        raise UploadError(
            f"only {len(frame)} rows have a non-missing target — at least "
            f"{MIN_ROWS} are needed for a meaningful train/holdout split."
        )

    derivation = derive_target(frame[TARGET_COLUMN])

    key = _upload_key(csv_bytes, target_column)
    existing = upload_manifest_path(key)
    if existing.exists():
        # Idempotent re-ingest: same bytes + same choice → the already-sealed
        # split stands. Re-splitting would mint a second holdout for the same
        # data, and two seals for one dataset is how honesty stories die.
        manifest = json.loads(existing.read_text(encoding="utf-8"))
        register_upload(manifest)
        manifest["already_ingested"] = True
        return manifest

    stratify = (frame[TARGET_COLUMN]
                if derivation.task == "classification" else None)
    try:
        train_df, holdout_df = train_test_split(
            frame, test_size=HOLDOUT_FRACTION, random_state=SEED,
            stratify=stratify,
        )
    except ValueError as exc:  # stratify with a 1-member class, etc.
        raise UploadError(f"could not build a stratified split: {exc}") from None
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    spec = DatasetSpec(
        key=key, openml_name=f"upload:{_safe_name(filename)}", version=0,
        task=derivation.task, subtype=derivation.subtype,
        metric=derivation.metric,
        note=f"User-uploaded CSV ({_safe_name(filename)}), target column "
             f"{target_column!r} chosen by the uploader; split sealed at "
             f"ingestion (Day 26).",
    )

    dataset_dir(key).mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path(key), index=False)
    holdout_df.to_parquet(holdout_path(key), index=False)

    manifest = {
        "key": key,
        "spec": spec_asdict(spec),
        "source": {
            "filename": filename,
            "source_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "target_column_as_uploaded": target_column,
            "n_rows_uploaded": int(n_before),
            "n_rows_dropped_missing_target": int(n_dropped),
        },
        "derivation": derivation.asdict(),
        "seed": SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "n_train": int(len(train_df)),
        "n_holdout": int(len(holdout_df)),
        "n_features": int(frame.shape[1] - 1),
        "train_sha256": sha256_of_frame(train_df),
        "holdout_sha256": sha256_of_frame(holdout_df),
        # Byte seals over the just-written files (Day 27): verifiable in any
        # environment, unlike the frame seals above (pandas/pyarrow-version
        # sensitive), which are kept for continuity with pre-Day-27 manifests.
        "train_file_sha256": sha256_of_file(train_path(key)),
        "holdout_file_sha256": sha256_of_file(holdout_path(key)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    upload_manifest_path(key).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    register_upload(manifest)
    manifest["already_ingested"] = False
    return manifest


def register_upload(manifest: dict[str, Any]) -> DatasetSpec:
    """(Re-)register an ingested upload in the live REGISTRY."""
    spec = DatasetSpec(**manifest["spec"])
    REGISTRY[spec.key] = spec
    return spec


def discover_upload_manifests() -> list[Path]:
    """Paths of every sealed upload manifest on disk (sorted; may be empty)."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        entry / UPLOAD_MANIFEST_NAME
        for entry in DATA_DIR.iterdir()
        if entry.is_dir() and is_upload_key(entry.name)
        and (entry / UPLOAD_MANIFEST_NAME).exists()
    )


def restore_uploaded_datasets() -> list[str]:
    """Re-register every sealed upload on disk (API restart survival).

    The REGISTRY is in-memory; uploads would otherwise 404 on /run after a
    restart even though their sealed splits are still on disk and verifiable.
    """
    restored: list[str] = []
    if not DATA_DIR.exists():
        return restored
    for entry in sorted(DATA_DIR.iterdir()):
        if not entry.is_dir() or not is_upload_key(entry.name):
            continue
        mpath = entry / UPLOAD_MANIFEST_NAME
        if not mpath.exists() or not (entry / "train.parquet").exists():
            continue
        try:
            register_upload(json.loads(mpath.read_text(encoding="utf-8")))
            restored.append(entry.name)
        except Exception:
            continue  # a broken upload dir must not take the API down
    return restored
