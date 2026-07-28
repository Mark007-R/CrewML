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
# Per-run LLM token cap and wall-clock cap, enforced by crewml.budget (Day 21):
# every llm.chat call is gated on the active RunBudget pre-call and charged to it
# after; the Critic finalises early when a cap is spent or unaffordable. <= 0
# means uncapped. (The token budget existed since Day 5 but was unenforced.)
RUN_TOKEN_BUDGET = int(os.getenv("CREWML_RUN_TOKEN_BUDGET", "200000"))
RUN_TIME_BUDGET_S = int(os.getenv("CREWML_RUN_TIME_BUDGET_S", "1800"))

# --- Executor sandbox (Day 19 hardening) ---
# On by default; CREWML_EXECUTOR_SANDBOX=0 is the explicit escape hatch.
EXECUTOR_SANDBOX = os.getenv("CREWML_EXECUTOR_SANDBOX", "1").lower() not in (
    "0", "false", "off",
)
# Child memory cap in MiB (0 = uncapped). POSIX: hard RLIMIT_AS; Windows: a
# parent-side watchdog kills the direct child when its working set exceeds it.
EXECUTOR_MEM_MB = int(os.getenv("CREWML_EXECUTOR_MEM_MB", "3072"))

# --- Self-repair loop (Day 20) ---
# Master switch: when generated code crashes, let the writing agent see the
# traceback and try again. Per-node overrides: CREWML_TRAINER_SELF_REPAIR /
# CREWML_FE_SELF_REPAIR. Off => the pre-Day-20 observe-and-degrade behaviour.
SELF_REPAIR = os.getenv("CREWML_SELF_REPAIR", "1").lower() not in (
    "0", "false", "off",
)
# Hard cap on repair attempts per failure (each attempt = one LLM call + one
# sandboxed re-run). Timeouts/OOM kills are never repaired regardless.
SELF_REPAIR_MAX_ATTEMPTS = int(os.getenv("CREWML_SELF_REPAIR_MAX_ATTEMPTS", "2"))

# Per-dataset wall-clock budget for the Day 4 classical-AutoML ceiling (FLAML).
# Held >= the crew's per-node executor timeout so beating AutoML is never an
# artifact of handing the crew more compute (EVAL_PROTOCOL.md §4).
AUTOML_TIME_BUDGET_S = int(os.getenv("CREWML_AUTOML_TIME_BUDGET_S", "120"))


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
