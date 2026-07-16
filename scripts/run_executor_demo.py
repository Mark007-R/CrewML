"""Day 6 demo — drive the sandboxed executor end-to-end on real (train-only) data.

Proves the crux tool works the way the crew will use it on Day 9: hand it a code
string + input files, get back a structured result with parsed metrics and the
artifacts the code produced. This mirrors a Trainer round-trip but stays honest —
it copies only the **train** split into the sandbox; the holdout is never named.

    python scripts/run_executor_demo.py --dataset credit-g

Also runs two negative cases (a crash and a timeout) so you can see the executor
report failure structurally instead of hanging or throwing.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml import executor
from crewml.datasets import REGISTRY, train_path

# A self-contained Trainer-style script: read the staged train split, fit a
# cross-validated model, emit the CV score + an artifact. No network, train-only.
TRAINER_CODE = textwrap.dedent(
    """\
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from crew_io import emit_metrics, artifact_path, input_path, SEED

    task = "__TASK__"
    metric = "__METRIC__"
    df = pd.read_parquet(input_path("train.parquet"))
    X, y = df.drop(columns=["target"]), df["target"]
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                           ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)],
        remainder="drop",
    )
    Model = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
    pipe = Pipeline([("pre", pre), ("model", Model(random_state=SEED))])
    scoring = {"roc_auc": "roc_auc", "macro_f1": "f1_macro", "r2": "r2"}[metric]
    scores = cross_val_score(pipe, X, y, cv=5, scoring=scoring)
    emit_metrics(cv_score=float(scores.mean()), cv_std=float(scores.std()), scoring=scoring, n=int(len(df)))

    pipe.fit(X, y)
    import joblib
    joblib.dump(pipe, artifact_path("model.joblib"))
    print(f"trained {Model.__name__}: {scoring}={scores.mean():.4f}")
    """
)

CRASH_CODE = "raise ValueError('deliberate boom — executor should report this, not raise')\n"
TIMEOUT_CODE = "import time\nwhile True:\n    time.sleep(1)\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="credit-g", choices=sorted(REGISTRY))
    args = ap.parse_args()

    spec = REGISTRY[args.dataset]
    code = TRAINER_CODE.replace("__TASK__", spec.task).replace("__METRIC__", spec.metric)

    print(f"=== executor demo · {spec.key} ({spec.task}, metric={spec.metric}) ===\n")

    print("[1/3] happy path — a Trainer-style CV fit on the TRAIN split only:")
    res = executor.run_code(code, inputs={"train.parquet": train_path(spec.key)})
    print(json.dumps(res.as_dict(), indent=2))
    print(f"stdout: {res.stdout.strip()}\n")
    assert res.ok, "trainer round-trip should succeed"
    assert "cv_score" in res.metrics, "metrics.json should carry cv_score"
    assert "model.joblib" in res.artifacts, "artifact should be collected"

    print("[2/3] failure path — a crash is reported, not raised:")
    crash = executor.run_code(CRASH_CODE)
    print(f"ok={crash.ok} returncode={crash.returncode} error={crash.error!r}\n")
    assert not crash.ok and "ValueError" in (crash.error or "")

    print("[3/3] timeout path — an infinite loop is killed at the cap (2s):")
    slow = executor.run_code(TIMEOUT_CODE, timeout_s=2)
    print(f"ok={slow.ok} timed_out={slow.timed_out} error={slow.error!r}\n")
    assert not slow.ok and slow.timed_out

    print("all three executor contracts verified [OK]")


if __name__ == "__main__":
    main()
