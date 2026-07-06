"""Central configuration: filesystem paths and environment settings.

Paths are resolved relative to the repo root so scripts work from any CWD.
Environment is read from a local ``.env`` if present (never committed).
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv optional at import time
    pass

# --- Filesystem layout (repo-root relative) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # git-ignored; raw + train/holdout splits live here
ARTIFACTS_DIR = ROOT / "artifacts"  # git-ignored; per-run outputs
RESULTS_DIR = ROOT / "results"      # committed; manifests, metrics, tables
REPORTS_DIR = ROOT / "reports"
EXPLAINERS_DIR = ROOT / "explainers"

for _d in (DATA_DIR, ARTIFACTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---
SEED = int(os.getenv("CREWML_SEED", "42"))
HOLDOUT_FRACTION = 0.2

# --- LLM provider ---
LLM_PROVIDER = os.getenv("CREWML_LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

# --- Run budgets ---
MAX_ITERATIONS = int(os.getenv("CREWML_MAX_ITERATIONS", "3"))
EXECUTOR_TIMEOUT_S = int(os.getenv("CREWML_EXECUTOR_TIMEOUT_S", "120"))
RUN_TOKEN_BUDGET = int(os.getenv("CREWML_RUN_TOKEN_BUDGET", "200000"))


def is_mock_mode() -> bool:
    """True when no usable LLM key is configured — pipeline runs offline.

    Mock-mode numbers must never be reported as real (see EVAL_PROTOCOL.md).
    """
    if LLM_PROVIDER == "mock":
        return True
    if LLM_PROVIDER == "groq":
        return not GROQ_API_KEY
    if LLM_PROVIDER in ("anthropic", "claude"):
        return not ANTHROPIC_API_KEY
    return True
