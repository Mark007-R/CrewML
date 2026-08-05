"""API client + pure helpers behind the Streamlit dashboard (Day 26).

Everything in this module runs headless — no Streamlit import — so the
dashboard's actual logic (what gets requested, how a trace or a derivation is
summarised, when the Run button may unlock) is unit-testable without a browser.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx
import pandas as pd

DEFAULT_API_URL = "http://127.0.0.1:8000"
TIMEOUT_S = 30.0  # /upload parses + splits + seals synchronously

# Display order/labels for the crew's node names as they appear in `trace`.
# Kept as plain data here (NOT imported from crewml.crew) so the dashboard
# stays a pure HTTP client with no crew imports.
CREW_NODE_ORDER: tuple[str, ...] = (
    "profiler", "planner", "feature_engineer", "trainer",
    "critic", "ensembler", "reporter",
)

NODE_LABELS: dict[str, str] = {
    "profiler": "Profiler — EDA & leakage screens",
    "planner": "Planner — modeling plan",
    "feature_engineer": "Feature Engineer — FE code",
    "trainer": "Trainer — CV training",
    "critic": "Critic — diagnose & decide",
    "ensembler": "Ensembler — combine models",
    "reporter": "Reporter — final report",
}

NODE_SHORT: dict[str, str] = {
    "profiler": "Profiler",
    "planner": "Planner",
    "feature_engineer": "Feature eng.",
    "trainer": "Trainer",
    "critic": "Critic",
    "ensembler": "Ensembler",
    "reporter": "Reporter",
}

TERMINAL_STATUSES = ("succeeded", "failed")


class ApiError(RuntimeError):
    """An API call that failed with a message fit to show in the UI."""


@dataclass
class CrewApiClient:
    """Small typed wrapper over the CrewML API routes the dashboard uses."""

    base_url: str = DEFAULT_API_URL

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        return self._request("GET", path, params=params or None)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        try:
            resp = httpx.request(method, url, timeout=TIMEOUT_S, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the CrewML API at {url}: {exc}") from None
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail")
            except Exception:
                detail = resp.text
            raise ApiError(f"{method} {path} -> {resp.status_code}: {detail}")
        return resp.json()

    # --- routes --------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._get("/healthz")

    def datasets(self) -> dict[str, dict[str, Any]]:
        return self._get("/datasets")["datasets"]

    def upload_csv(self, csv_bytes: bytes, *, filename: str,
                   target_column: str) -> dict[str, Any]:
        return self._request(
            "POST", "/upload",
            files={"file": (filename, csv_bytes, "text/csv")},
            data={"target_column": target_column},
        )

    def submit_run(self, dataset_key: str, **options: Any) -> str:
        body = {"dataset_key": dataset_key,
                **{k: v for k, v in options.items() if v is not None}}
        return self._request("POST", "/run", json=body)["run_id"]

    def status(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/status/{run_id}")

    def report(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/report/{run_id}")

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._get("/runs", limit=limit)["runs"]

    def metrics(self) -> dict[str, Any]:
        return self._get("/metrics")


# --- pure helpers (no I/O) ---------------------------------------------------

def column_options(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column facts for the target picker: name, dtype, distinct count.

    This feeds a dropdown the USER picks from — it deliberately computes no
    ranking, no "likely target" score, no default. Guessing the target is the
    bug the Day-26 spec forbids; the picker starts empty.
    """
    return [
        {"name": str(col), "dtype": str(df[col].dtype),
         "n_unique": int(df[col].nunique()), "n_missing": int(df[col].isna().sum())}
        for col in df.columns
    ]


def format_column_option(opt: dict[str, Any]) -> str:
    return (f"{opt['name']}  ({opt['dtype']}, {opt['n_unique']} distinct"
            + (f", {opt['n_missing']} missing" if opt["n_missing"] else "") + ")")


def derivation_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Flatten an upload manifest into the facts the confirm screen shows."""
    d = manifest.get("derivation") or {}
    src = manifest.get("source") or {}
    return {
        "dataset_key": manifest.get("key"),
        "task": d.get("task"),
        "subtype": d.get("subtype"),
        "metric": d.get("metric"),
        "rule": d.get("rule"),
        "warnings": d.get("warnings") or [],
        "target_column": src.get("target_column_as_uploaded"),
        "n_rows_uploaded": src.get("n_rows_uploaded"),
        "n_rows_dropped_missing_target": src.get("n_rows_dropped_missing_target"),
        "n_train": manifest.get("n_train"),
        "n_holdout": manifest.get("n_holdout"),
        "holdout_sha256": manifest.get("holdout_sha256"),
        "train_sha256": manifest.get("train_sha256"),
        "already_ingested": manifest.get("already_ingested", False),
    }


def trace_rows(progress: Optional[dict[str, Any]]) -> list[dict[str, str]]:
    """Render a progress snapshot's trace as (step, node, label) rows."""
    trace = (progress or {}).get("trace") or []
    return [
        {"step": str(i + 1), "node": node,
         "label": NODE_LABELS.get(node, node)}
        for i, node in enumerate(trace)
    ]


def node_states(progress: Optional[dict[str, Any]],
                *, finished: bool = False) -> list[dict[str, Any]]:
    """Per-node pipeline state for the live-trace visual, from a trace snapshot.

    Returns one entry per node in ``CREW_NODE_ORDER``:
    ``{"node", "label", "state", "visits"}`` where state is ``"done"``,
    ``"active"`` (the last-visited node of an unfinished run) or ``"pending"``.
    ``visits`` counts revisits — the Critic loop makes planner/fe/trainer/critic
    legitimately appear more than once, and hiding that would hide the loop.
    """
    trace = (progress or {}).get("trace") or []
    visits = {n: trace.count(n) for n in CREW_NODE_ORDER}
    current = trace[-1] if trace and not finished else None
    out = []
    for n in CREW_NODE_ORDER:
        if n == current:
            state = "active"
        elif visits[n]:
            state = "done"
        else:
            state = "pending"
        out.append({"node": n, "label": NODE_SHORT[n],
                    "state": state, "visits": visits[n]})
    return out


def is_finished(status: dict[str, Any]) -> bool:
    return status.get("status") in TERMINAL_STATUSES


def run_label(status: dict[str, Any]) -> str:
    """One-line run descriptor for pickers: id, dataset, state, score if any."""
    head = status.get("headline") or {}
    score = head.get("final_cv_score")
    bits = [status.get("run_id", "?"), status.get("dataset_key", "?"),
            status.get("status", "?")]
    if score is not None:
        bits.append(f"cv={score:.4f}")
    return " · ".join(bits)
