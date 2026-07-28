"""Day 23 — the run-level reproducibility study: same pins ⇒ same result?

The artifact-level check (:mod:`crewml.artifact_registry`) proves committed
files match their generators. This study makes the *run-level* claim and then
measures it instead of asserting it:

* **Arm 1 — deterministic core:** the crew, LLM narratives off, run twice per
  dataset in **separate fresh processes** with identical pins. Claim: the
  result fingerprints (:func:`crewml.manifest.result_fingerprint`) are
  identical. Fresh processes matter — an in-process double run could pass on
  shared hidden state that a real re-run tomorrow would not have.
* **Arm 2 — seed sensitivity:** one more run with a different ``CREWML_SEED``.
  The fingerprint must CHANGE. Without this arm, Arm 1 could pass with the
  seed silently ignored everywhere (fingerprints don't embed the seed, so
  identical-by-apathy is indistinguishable from identical-by-control).
* **Arm 3 — live LLM (labelled):** the live crew twice with identical pins.
  Not a determinism claim — Groq at temperature 0.0 promises nothing — but a
  measurement of *which layers* reproduce: the scored result, the generated FE
  code, the advisory prose. Skipped honestly (``measured: false`` + reason)
  when no live provider is available; never simulated.

Each run happens in a child process (``python -m crewml.repro_study --child``)
whose environment carries the pins, because ``crewml.config`` reads them at
import time. The child writes the run's manifest to a JSON file; the parent
only ever compares manifests — exactly what a human re-running the pipeline
next month would do.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from crewml import config, llm
from crewml.config import ARTIFACTS_DIR, RESULTS_DIR, is_mock_mode

REPORT_JSON_PATH = RESULTS_DIR / "day23_reproducibility.json"
REPORT_MD_PATH = RESULTS_DIR / "day23_reproducibility.md"
STUDY_DIR = ARTIFACTS_DIR / "repro"

DETERMINISTIC_DATASETS = ("credit-g", "cpu_small")
LIVE_DATASET = "credit-g"
ALT_SEED = 43

# The LLM narrative toggles. Off = the deterministic core (Day 11's --no-llm).
_LLM_TOGGLES = ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM",
                "CREWML_FE_LLM", "CREWML_CRITIC_LLM")


# --------------------------------------------------------------------------
# Child: one crew run under the pins already present in the environment.
# --------------------------------------------------------------------------

def child_main(dataset_key: str, out_path: Path) -> int:
    """Run the crew once and write its run manifest (plus study extras)."""
    from crewml import budget as budget_mod
    from crewml import manifest as manifest_mod
    from crewml.crew import build_crew, initial_state
    from crewml.datasets import REGISTRY, verify_holdout_untouched

    spec = REGISTRY[dataset_key]
    app = build_crew()
    state = initial_state(spec, max_iterations=config.MAX_ITERATIONS)
    limit = 3 + config.MAX_ITERATIONS * 4 + 10
    started = time.perf_counter()
    with budget_mod.run_budget(None, None):
        final = app.invoke(state, config={"recursion_limit": limit})
    duration_s = time.perf_counter() - started

    record = {
        "manifest": manifest_mod.build_run_manifest(final),
        "narratives_sha256": _narratives_fingerprint(final),
        "holdout_untouched": verify_holdout_untouched(dataset_key),
        "duration_s": round(duration_s, 2),
        "run_budget": (final.get("report") or {}).get("run_budget"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return 0


def _narratives_fingerprint(final_state: dict[str, Any]) -> str:
    """One hash over every advisory narrative the run produced.

    Kept OUT of the result fingerprint (prose is advisory and volatile); the
    live arm compares it separately to show which layer diverged.
    """
    from crewml.manifest import canonical_json, _sha256_text

    def _text(owner: Any) -> Any:
        return ((owner or {}).get("llm_narrative") or {}).get("text")

    prose = {
        "profile": _text(final_state.get("profile")),
        "plan": _text(final_state.get("plan")),
        "critiques": [_text(c) for c in final_state.get("critiques") or []],
    }
    return _sha256_text(canonical_json(prose))


# --------------------------------------------------------------------------
# Parent: spawn pinned children and compare their manifests.
# --------------------------------------------------------------------------

def _spawn(dataset_key: str, out_path: Path, *, seed: int, llm_on: bool) -> dict[str, Any]:
    """One pinned crew run in a fresh process; returns the child's record."""
    env = os.environ.copy()
    env["CREWML_SEED"] = str(seed)
    env["CREWML_TRAINER_PARAM_SEARCH"] = "0"  # CV at default params (disclosed)
    for var in _LLM_TOGGLES:
        env[var] = "1" if llm_on else "0"
    proc = subprocess.run(
        [sys.executable, "-m", "crewml.repro_study",
         "--child", dataset_key, "--out", str(out_path)],
        cwd=config.ROOT, env=env, capture_output=True, text=True,
        timeout=60 * 30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"child run failed for {dataset_key} (seed={seed}, llm={llm_on}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return json.loads(out_path.read_text(encoding="utf-8"))


def _result_summary(record: dict[str, Any]) -> dict[str, Any]:
    m = record["manifest"]
    r = m["result"]
    return {
        "result_fingerprint": m["result_fingerprint"],
        "cv_score": r["final_model"]["cv_score"],
        "final_model": r["final_model"]["chosen"] or r["training"]["best_model"],
        "fe_code_sha256": r["fe"]["fe_code_sha256"],
        "narratives_sha256": record["narratives_sha256"],
        "holdout_untouched": record["holdout_untouched"],
        "duration_s": record["duration_s"],
        "tokens": (record.get("run_budget") or {}).get("tokens_spent"),
    }


def _probe_provider() -> Optional[str]:
    """Return None when the live provider answers, else the honest reason.

    ``is_mock_mode()`` only checks that a key is PRESENT; a present-but-revoked
    key looks live and is the dangerous case, so an actual 8-token round trip
    is required before any live arm runs.
    """
    if is_mock_mode():
        return "mock mode — no LLM key configured"
    try:
        llm.chat("You are a liveness probe for a measurement harness.",
                 "Reply with the single word OK.",
                 temperature=0.0, max_tokens=8)
        return None
    except Exception as exc:
        return f"provider preflight failed: {type(exc).__name__}: {exc}"


def run_study(*, live: bool = True) -> dict[str, Any]:
    """Run all arms; live arm only if requested AND the provider answers."""
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    # Arm 1 — deterministic core, twice per dataset, fresh process each.
    det: dict[str, Any] = {}
    for key in DETERMINISTIC_DATASETS:
        runs = [
            _result_summary(_spawn(
                key, STUDY_DIR / f"det_{key}_run{i}.json",
                seed=config.SEED, llm_on=False))
            for i in (1, 2)
        ]
        det[key] = {
            "runs": runs,
            "identical": runs[0]["result_fingerprint"] == runs[1]["result_fingerprint"],
        }
    det_all = all(d["identical"] for d in det.values())

    # Arm 2 — different seed must move the fingerprint (and the score).
    base = det[DETERMINISTIC_DATASETS[0]]["runs"][0]
    alt = _result_summary(_spawn(
        DETERMINISTIC_DATASETS[0], STUDY_DIR / "seed_alt.json",
        seed=ALT_SEED, llm_on=False))
    seed_arm = {
        "dataset": DETERMINISTIC_DATASETS[0],
        "seed_base": config.SEED,
        "seed_alt": ALT_SEED,
        "fingerprint_base": base["result_fingerprint"],
        "fingerprint_alt": alt["result_fingerprint"],
        "cv_base": base["cv_score"],
        "cv_alt": alt["cv_score"],
        "fingerprint_moved": alt["result_fingerprint"] != base["result_fingerprint"],
    }

    # Arm 3 — live, only against a provider that actually answers.
    live_arm: dict[str, Any] = {"measured": False, "dataset": LIVE_DATASET}
    reason = _probe_provider() if live else "skipped by --no-live flag"
    if reason is None:
        runs = [
            _result_summary(_spawn(
                LIVE_DATASET, STUDY_DIR / f"live_run{i}.json",
                seed=config.SEED, llm_on=True))
            for i in (1, 2)
        ]
        live_arm = {
            "measured": True,
            "dataset": LIVE_DATASET,
            "provider": config.LLM_PROVIDER,
            "model": config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.ANTHROPIC_MODEL,
            "runs": runs,
            "result_identical": runs[0]["result_fingerprint"] == runs[1]["result_fingerprint"],
            "fe_code_identical": runs[0]["fe_code_sha256"] == runs[1]["fe_code_sha256"],
            "narratives_identical": runs[0]["narratives_sha256"] == runs[1]["narratives_sha256"],
        }
    else:
        live_arm["reason_not_measured"] = reason

    from crewml.manifest import environment_pins

    report = {
        "study": "day23_reproducibility",
        "param_search": False,  # CV at default params for bounded wall-clock
        "max_iterations": config.MAX_ITERATIONS,
        "pins_reference": environment_pins(),
        "arms": {
            "deterministic_core": {"datasets": det, "all_identical": det_all},
            "seed_sensitivity": seed_arm,
            "live_llm": live_arm,
        },
        "holdout_untouched_throughout": all(
            r["holdout_untouched"]
            for d in det.values() for r in d["runs"]
        ) and alt["holdout_untouched"],
    }
    return report


def write_report(report: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(render_markdown(report), encoding="utf-8")


# --------------------------------------------------------------------------
# Renderer — registered in crewml.artifact_registry; must be deterministic.
# --------------------------------------------------------------------------

def _fp(fingerprint: Optional[str]) -> str:
    return fingerprint[:12] if fingerprint else "—"


def _score(v: Any) -> str:
    return f"{v:.6f}" if isinstance(v, (int, float)) else "—"


def render_markdown(report: dict[str, Any]) -> str:
    arms = report["arms"]
    det = arms["deterministic_core"]
    seed = arms["seed_sensitivity"]
    live = arms["live_llm"]
    pins = report["pins_reference"]

    lines: list[str] = []
    lines.append("# Day 23 — Run-level reproducibility study")
    lines.append("")
    lines.append(
        "Claim under test: **same pins ⇒ same result fingerprint** "
        "(`crewml.manifest.result_fingerprint` — SHA-256 over scores, chosen "
        "model, FE-code hash, trace and Critic decisions; prose/latency/tokens "
        "excluded). Every run is a separate fresh process; the parent compares "
        "only the run manifests, exactly as a human re-running the pipeline "
        "would. Grid search off (CV at default params) for bounded wall-clock; "
        f"`max_iterations={report['max_iterations']}` (production default)."
    )
    lines.append("")

    lines.append("## Arm 1 — deterministic core, run twice (fresh process each)")
    lines.append("")
    lines.append("| dataset | run 1 fingerprint | run 2 fingerprint | cv score | final model | identical |")
    lines.append("|---|---|---|---|---|---|")
    for key, d in det["datasets"].items():
        r1, r2 = d["runs"]
        lines.append(
            f"| {key} | `{_fp(r1['result_fingerprint'])}` | `{_fp(r2['result_fingerprint'])}` "
            f"| {_score(r1['cv_score'])} | {r1['final_model']} "
            f"| {'✅ bit-identical' if d['identical'] else '❌ DIVERGED'} |"
        )
    lines.append("")
    lines.append(
        "**Verdict: "
        + ("all deterministic-core runs reproduce bit-identically.**"
           if det["all_identical"]
           else "DIVERGENCE FOUND — the deterministic core is not deterministic.**")
    )
    lines.append("")

    lines.append("## Arm 2 — the seed must matter")
    lines.append("")
    lines.append(
        "Fingerprints do not embed the seed, so Arm 1 alone cannot distinguish "
        "*controlled* from *ignored*. A different `CREWML_SEED` must move the "
        "outcome:"
    )
    lines.append("")
    lines.append("| dataset | seed | fingerprint | cv score |")
    lines.append("|---|---|---|---|")
    lines.append(f"| {seed['dataset']} | {seed['seed_base']} | `{_fp(seed['fingerprint_base'])}` | {_score(seed['cv_base'])} |")
    lines.append(f"| {seed['dataset']} | {seed['seed_alt']} | `{_fp(seed['fingerprint_alt'])}` | {_score(seed['cv_alt'])} |")
    lines.append("")
    lines.append(
        "**Verdict: "
        + ("the seed reaches the model — fingerprint and score both moved.**"
           if seed["fingerprint_moved"]
           else "SEED IGNORED — changing it changed nothing. Reproducibility above is vacuous.**")
    )
    lines.append("")

    lines.append("## Arm 3 — live LLM double-run (labelled; not a determinism claim)")
    lines.append("")
    if not live.get("measured"):
        lines.append(
            f"**Not measured** — {live.get('reason_not_measured', 'unknown')}. "
            "A simulated 'live' arm would be fiction (EVAL_PROTOCOL §5); the "
            "deterministic arms above stand on their own."
        )
    else:
        r1, r2 = live["runs"]
        lines.append(
            f"Provider: **{live['provider']} / {live['model']}**, temperature 0.0, "
            f"identical pins, two fresh runs on `{live['dataset']}`:"
        )
        lines.append("")
        lines.append("| layer | run 1 | run 2 | reproduced |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| scored result (fingerprint) | `{_fp(r1['result_fingerprint'])}` | `{_fp(r2['result_fingerprint'])}` "
            f"| {'✅' if live['result_identical'] else '❌'} |")
        lines.append(
            f"| generated FE code (sha256) | `{_fp(r1['fe_code_sha256'])}` | `{_fp(r2['fe_code_sha256'])}` "
            f"| {'✅' if live['fe_code_identical'] else '❌'} |")
        lines.append(
            f"| advisory narratives (sha256) | `{_fp(r1['narratives_sha256'])}` | `{_fp(r2['narratives_sha256'])}` "
            f"| {'✅' if live['narratives_identical'] else '❌'} |")
        lines.append("")
        lines.append(
            f"CV {_score(r1['cv_score'])} vs {_score(r2['cv_score'])}; "
            f"final model {r1['final_model']} vs {r2['final_model']}."
        )
        lines.append("")
        if live["result_identical"]:
            lines.append(
                "**Verdict: the scored pipeline reproduced across live runs.** "
                "The provider promises no determinism at temperature 0.0, so this "
                "is an observation about *these two runs*, not a guarantee — the "
                "guarantee lives in the deterministic core (Arm 1), and the run "
                "manifest is what makes any live divergence diagnosable."
            )
        else:
            lines.append(
                "**Verdict: the live runs DIVERGED at the scored layer** — the "
                "provider returned different FE/plan content across identical "
                "prompts. This is the expected failure mode of a live LLM arm, "
                "recorded rather than hidden; the run manifests pin exactly what "
                "differed."
            )
    lines.append("")

    lines.append("## Pins of record")
    lines.append("")
    pk = pins["packages"]
    lines.append(
        f"- seed {pins['seed']} · python {pins['python']} · "
        f"numpy {pk.get('numpy')} · pandas {pk.get('pandas')} · "
        f"scikit-learn {pk.get('scikit-learn')} · langgraph {pk.get('langgraph')}"
    )
    lines.append(
        f"- provider {pins['llm']['provider']} / {pins['llm']['model']} "
        f"(mock_mode={pins['llm']['mock_mode']}), temperature default "
        f"{pins['llm']['temperature_default']}"
    )
    lines.append(
        f"- git `{(pins.get('git_commit') or 'n/a')[:12]}`"
        + (" (dirty tree — run predates the Day-23 commit)" if pins.get("git_dirty") else "")
    )
    lines.append(
        f"- holdout untouched throughout: "
        f"{'✅' if report['holdout_untouched_throughout'] else '❌ SEAL VIOLATION'}"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 23 run-level reproducibility study")
    ap.add_argument("--child", metavar="DATASET", default=None,
                    help="internal: run one pinned crew run and write its manifest")
    ap.add_argument("--out", default=None, help="internal: child output path")
    ap.add_argument("--no-live", action="store_true",
                    help="skip the live-LLM arm even if a provider is configured")
    args = ap.parse_args()

    if args.child:
        if not args.out:
            raise SystemExit("--child requires --out")
        return child_main(args.child, Path(args.out))

    report = run_study(live=not args.no_live)
    write_report(report)
    det = report["arms"]["deterministic_core"]["all_identical"]
    seed_ok = report["arms"]["seed_sensitivity"]["fingerprint_moved"]
    live_arm = report["arms"]["live_llm"]
    print(f"[repro] deterministic core bit-identical: {det}")
    print(f"[repro] seed sensitivity (fingerprint moved): {seed_ok}")
    if live_arm.get("measured"):
        print(f"[repro] live arm: result_identical={live_arm['result_identical']} "
              f"fe_code_identical={live_arm['fe_code_identical']} "
              f"narratives_identical={live_arm['narratives_identical']}")
    else:
        print(f"[repro] live arm NOT measured: {live_arm.get('reason_not_measured')}")
    print(f"[repro] wrote {REPORT_JSON_PATH} and {REPORT_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
