"""Merge two self-repair study passes into one board — only if they are comparable.

    python scripts/merge_self_repair_passes.py BASE.json ADDITION.json [--out PATH]

Day 20 measured its 9 fault families in two live passes: the eight invented faults
first, then ``non_finite`` once Groq's daily token budget reset. Combining two runs
into one headline is exactly the kind of step that quietly launders an incomparable
number, so this script refuses unless the passes agree on everything that could move
a score, and it recomputes every summary field from the merged ``runs`` array rather
than trusting either input's totals.

What it asserts before merging:

* same ``repairer`` mode, ``provider`` and ``model``
* same ``max_attempts`` budget
* **bit-identical clean-control scores** per shared dataset — the strongest available
  evidence that seeds, splits and the deterministic core did not drift between passes

What it deliberately does NOT claim: that both passes ran identical *code*. Day 20's
pass 1 predates commit ``e6685e7`` (which tightened the repair acceptance gate), so
``code_note`` records that skew in the output instead of papering over it. A
reproducibility-grade merge would re-run every pass under one commit; this records
what actually happened.

Only controls are deduplicated — injected runs from the addition are appended, so
merging a pass that re-measures an existing fault would produce duplicate rows on
purpose, and the operator has to notice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.self_repair_study import NON_RESTORABLE_FAULTS, SELF_REPAIR_RESULT_PATH


def assert_comparable(base: dict, add: dict) -> None:
    """Refuse to merge passes that could differ in anything score-bearing."""
    for field in ("repairer", "provider", "model", "max_attempts"):
        if base.get(field) != add.get(field):
            raise SystemExit(
                f"refusing to merge: {field} differs "
                f"({base.get(field)!r} vs {add.get(field)!r})"
            )
    shared = set(base.get("clean_scores", {})) & set(add.get("clean_scores", {}))
    if not shared:
        raise SystemExit("refusing to merge: no shared dataset to compare controls on")
    for ds in sorted(shared):
        b, a = base["clean_scores"][ds], add["clean_scores"][ds]
        if b != a:
            raise SystemExit(
                f"refusing to merge: clean-control score for {ds} differs "
                f"({b} vs {a}) — the passes are not comparable"
            )
    print(f"[merge] comparability OK; controls identical on {sorted(shared)}")


def merge(base: dict, add: dict, *, code_note: str | None = None) -> dict:
    assert_comparable(base, add)

    added = [r for r in add["runs"] if r["fault"] != "none_control"]
    merged = base["runs"] + added
    out = dict(base)

    inj = [r for r in merged if r["fault"] != "none_control"]
    ctl = [r for r in merged if r["fault"] == "none_control"]
    unmeasured = [r for r in inj if r.get("llm_unavailable")]
    measurable = [r for r in inj if not r.get("llm_unavailable")]
    rec = [r for r in measurable if r["recovered"]]
    fid = [
        abs(r["score_fidelity_vs_clean"])
        for r in rec
        if r["score_fidelity_vs_clean"] is not None and r.get("restorable", True)
    ]
    nonrest = [
        {"dataset": r["dataset"], "fault": r["fault"],
         "delta": r["score_fidelity_vs_clean"]}
        for r in rec
        if not r.get("restorable", True) and r["score_fidelity_vs_clean"] is not None
    ]

    out.update(
        runs=merged,
        n_faults=len({r["fault"] for r in inj}),
        n_injected_runs=len(inj),
        measurable_runs=len(measurable),
        unmeasured_runs=len(unmeasured),
        unmeasured_faults=sorted({r["fault"] for r in unmeasured}),
        measurement_valid=len(measurable) > 0,
        measurement_caveat=None if not unmeasured else (
            f"{len(unmeasured)} of {len(inj)} injected runs never reached the provider; "
            "excluded from the recovery-rate denominator."
        ),
        recovery_rate=round(len(rec) / len(measurable), 4) if measurable else None,
        recovered_runs=len(rec),
        first_attempt_recoveries=sum(1 for r in rec if r["recovered_on_attempt"] == 1),
        mean_abs_score_fidelity=round(sum(fid) / len(fid), 6) if fid else None,
        max_abs_score_fidelity=round(max(fid), 6) if fid else None,
        n_fidelity_scored=len(fid),
        non_restorable_faults=sorted(NON_RESTORABLE_FAULTS),
        non_restorable_deltas=nonrest,
        fe_artifact_inconsistencies=sum(
            1 for r in rec if r["fe_artifact_consistent"] is False
        ),
        false_positive_repairs_on_clean=sum(1 for r in ctl if r["repair_attempted"]),
        total_prompt_tokens=base["total_prompt_tokens"] + add["total_prompt_tokens"],
        total_completion_tokens=(
            base["total_completion_tokens"] + add["total_completion_tokens"]
        ),
        total_wall_s=round(base["total_wall_s"] + sum(r["wall_s"] for r in added), 1),
        holdout_seal_intact=bool(
            base["holdout_seal_intact"] and add["holdout_seal_intact"]
        ),
        merged_from_passes=2,
        code_note=code_note,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", type=Path, help="the earlier pass's JSON")
    ap.add_argument("addition", type=Path, help="the later pass's JSON")
    ap.add_argument("--out", type=Path, default=SELF_REPAIR_RESULT_PATH)
    ap.add_argument(
        "--code-note", default=None,
        help="record any code skew between the passes rather than hiding it",
    )
    args = ap.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    add = json.loads(args.addition.read_text(encoding="utf-8"))
    out = merge(base, add, code_note=args.code_note)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"[merge] {out['recovered_runs']}/{out['measurable_runs']} "
        f"= {out['recovery_rate']:.0%} across {out['n_faults']} faults "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
