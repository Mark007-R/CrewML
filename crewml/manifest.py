"""Per-run manifest + result fingerprint — the Day-23 reproducibility record.

A crew run is only *claimably* reproducible if the claim names what was held
fixed and what the outcome was. This module produces both halves:

* :func:`build_run_manifest` — everything that pins a run: the seed, the split
  seals it ran against, the package/interpreter versions, the provider and
  mock-mode flag, every ``CREWML_*`` knob in the environment, and the git
  commit. If two manifests' ``pins`` differ, nobody promised the same result.
* :func:`result_fingerprint` — a SHA-256 over the *deterministic outcome* of a
  finished run (scores, chosen model, FE code hash, node trace, Critic
  decisions). Volatile facts — wall-clock, token counts, LLM narrative prose,
  artifact paths, run ids — are deliberately excluded: they legitimately differ
  between two honest runs of the same pinned configuration.

The reproducibility contract this enables (tested in
``tests/test_reproducibility.py``, measured in ``crewml/repro_study.py``):
**same pins ⇒ same result fingerprint** for the deterministic core. Live-LLM
runs get the honest, weaker statement: the fingerprint tells you *whether* the
outcome reproduced; the manifest tells you what was pinned when it didn't.

Secrets never enter a manifest: environment capture is allowlisted to the
``CREWML_`` prefix, and API keys live under ``GROQ_``/``ANTHROPIC_`` names.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from crewml import config

MANIFEST_SCHEMA_VERSION = 1

# Libraries whose versions can move a score. flaml is optional (Day-4 ceiling
# only); a missing package records as None rather than failing the manifest.
_PINNED_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "scipy", "langgraph", "matplotlib",
    "flaml",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable serialisation for hashing: sorted keys, no whitespace drift."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _package_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in _PINNED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_commit() -> Optional[str]:
    """Current HEAD, or None outside a repo / without git — never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    """True when the working tree differs from HEAD (uncommitted code ran).

    A manifest whose ``git_commit`` names a commit the run's code doesn't match
    would be a false pin — this flag keeps it honest. None if git is absent.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.ROOT, capture_output=True, text=True, timeout=10,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def _crewml_env() -> dict[str, str]:
    """Every CREWML_* variable set in the environment (allowlist by prefix).

    This is the whole knob surface — LLM toggles, search toggle, budgets,
    sandbox settings — and by construction cannot contain an API key.
    """
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("CREWML_")}


def dataset_seals(dataset_key: str) -> dict[str, Any]:
    """The locked split identity this run stands on.

    Benchmark keys read the committed Day-1 manifest; uploaded datasets
    (Day 26, ``upload-*``) read the per-upload manifest sealed at ingestion —
    same shape, so a run manifest over user data carries the same seal
    evidence. Lazy import: :mod:`crewml.uploads` imports the datasets module.
    """
    if dataset_key.startswith("upload-"):
        from crewml.uploads import load_upload_manifest

        entry = load_upload_manifest(dataset_key)
        seals = {
            "split_seed": entry["seed"],
            "train_sha256": entry["train_sha256"],
            "holdout_sha256": entry["holdout_sha256"],
            "n_train": entry["n_train"],
            "n_holdout": entry["n_holdout"],
            "sealed_at_ingestion": True,
        }
        # Day-27 byte seals, when the manifest has them (pre-Day-27 uploads
        # legitimately don't — verification falls back to the frame seal).
        for k in ("train_file_sha256", "holdout_file_sha256"):
            if entry.get(k):
                seals[k] = entry[k]
        return seals
    manifest_path = config.RESULTS_DIR / "dataset_manifest.json"
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = recorded["datasets"][dataset_key]
    seals = {
        "split_seed": recorded["seed"],
        "train_sha256": entry["train_sha256"],
        "holdout_sha256": entry["holdout_sha256"],
        "n_train": entry["n_train"],
        "n_holdout": entry["n_holdout"],
    }
    for k in ("train_file_sha256", "holdout_file_sha256"):
        if entry.get(k):
            seals[k] = entry[k]
    return seals


def environment_pins() -> dict[str, Any]:
    """Everything held fixed *around* a run (no run outcome in here)."""
    return {
        "seed": config.SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "llm": {
            "provider": config.LLM_PROVIDER,
            "model": (config.GROQ_MODEL if config.LLM_PROVIDER == "groq"
                      else config.ANTHROPIC_MODEL),
            "mock_mode": config.is_mock_mode(),
            # llm.chat pins temperature=0.0 by default; recorded so a future
            # default change shows up as a pin difference, not a mystery.
            "temperature_default": 0.0,
        },
        "config": {
            "max_iterations": config.MAX_ITERATIONS,
            "executor_timeout_s": config.EXECUTOR_TIMEOUT_S,
            "executor_sandbox": config.EXECUTOR_SANDBOX,
            "executor_mem_mb": config.EXECUTOR_MEM_MB,
            "self_repair": config.SELF_REPAIR,
            "self_repair_max_attempts": config.SELF_REPAIR_MAX_ATTEMPTS,
            "run_token_budget": config.RUN_TOKEN_BUDGET,
            "run_time_budget_s": config.RUN_TIME_BUDGET_S,
        },
        "crewml_env": _crewml_env(),
    }


def canonical_result(final_state: dict[str, Any]) -> dict[str, Any]:
    """The deterministic outcome of a finished crew run, and nothing else.

    Every field here either IS the result (scores, chosen model) or determines
    it (FE code, decisions, visit order). Anything that can differ between two
    honest runs of the same pins — latency, tokens, narrative prose, run ids,
    artifact paths, the budget ledger — is excluded on purpose; putting one in
    would make the fingerprint cry wolf.
    """
    training = final_state.get("training") or {}
    fe_meta = final_state.get("fe_meta") or {}
    report = final_state.get("report") or {}
    final_model = report.get("final_model") or {}
    critiques = final_state.get("critiques") or []
    fe_code = final_state.get("fe_code")

    return {
        "dataset_key": final_state.get("dataset_key"),
        "task": final_state.get("task"),
        "subtype": final_state.get("subtype"),
        "metric": final_state.get("metric"),
        "iterations_run": final_state.get("iteration"),
        "trace": final_state.get("trace"),
        "decisions": [c.get("decision") for c in critiques],
        "fe": {
            "source": fe_meta.get("source"),
            "fe_code_sha256": _sha256_text(fe_code) if fe_code else None,
        },
        "training": {
            "ok": training.get("ok"),
            "best_model": training.get("best_model"),
            "cv_score": training.get("cv_score"),
            "param_search": training.get("param_search"),
            "repaired": training.get("repaired"),
        },
        "final_model": {
            "kind": final_model.get("kind"),
            "chosen": final_model.get("chosen"),
            "members": final_model.get("members"),
            "single_best_model": final_model.get("single_best_model"),
            "cv_score": final_model.get("cv_score"),
            "ensemble_cv_score": final_model.get("ensemble_cv_score"),
            "single_best_cv_score": final_model.get("single_best_cv_score"),
        },
        "cv_score_is_holdout": False,
    }


def result_fingerprint(final_state: dict[str, Any]) -> str:
    """SHA-256 of the canonical result — equal fingerprints ⇔ same outcome."""
    return _sha256_text(canonical_json(canonical_result(final_state)))


def build_run_manifest(final_state: dict[str, Any]) -> dict[str, Any]:
    """The full record for one finished crew run: pins + outcome + fingerprint."""
    dataset_key = final_state.get("dataset_key")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": {
            "key": dataset_key,
            "seals": dataset_seals(dataset_key) if dataset_key else None,
        },
        "pins": environment_pins(),
        "result": canonical_result(final_state),
        "result_fingerprint": result_fingerprint(final_state),
    }


def write_run_manifest(final_state: dict[str, Any], path: Path) -> dict[str, Any]:
    """Build the manifest and persist it next to the run's other artifacts."""
    manifest = build_run_manifest(final_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
