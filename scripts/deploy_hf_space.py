"""Day 30 — assemble and deploy the CrewML Hugging Face Space.

    python scripts/deploy_hf_space.py --dry-run          # assemble + scan only
    python scripts/deploy_hf_space.py                    # deploy (no secrets)
    python scripts/deploy_hf_space.py --set-secret       # + GROQ_API_KEY from local .env
    python scripts/deploy_hf_space.py --wait 900         # poll until RUNNING/ERROR

The Space (``IamBatman07/CrewML``, sdk: docker, app_port: 7860) is a separate
git repo from GitHub, so this script does NOT push the working tree — it
assembles a staging directory with exactly what the Space needs and uploads
that:

* ``deploy/hf_space/`` files at the staging ROOT (Dockerfile, Space-card
  README, start.sh) — Spaces read the Dockerfile and card from the repo root.
* ``crewml/``, ``scripts/``, ``results/``, ``requirements.txt`` — the code and
  the committed evidence, minus caches.
* ``data/``: ONLY the five sealed benchmark splits + the demo CSV. The
  ``probe_*``/``upload-*`` fixtures are host-local study leftovers, and the
  GitHub repo ignores data/ entirely — but a Space has no host volume, so the
  splits ride in the Space repo and are verified against the byte seals in
  ``results/dataset_manifest.json`` at runtime like everywhere else.

Two guards run before any upload and hard-fail the deploy:

* secret scan — no provider-key VALUE patterns anywhere in the staging tree
  (variable NAMES like GROQ_API_KEY appear in code legitimately; what must
  never ship is a value: ``gsk_…``, ``sk-ant-…``, ``sk-proj-…``);
* .env exclusion — no dotenv file makes it into staging at all.

``--set-secret`` provisions the GROQ_API_KEY *Space secret* (runtime-env only,
never in the image or repo) from the local ``.env``, so the Space runs live
Groq; without it the Space boots in labelled mock mode.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.config import ROOT  # noqa: E402

SPACE_ID = "IamBatman07/CrewML"
SPACE_URL = "https://huggingface.co/spaces/IamBatman07/CrewML"
DEPLOY_SRC = ROOT / "deploy" / "hf_space"

BENCHMARK_DATASETS = ("credit-g", "diabetes", "vehicle", "cpu_small", "kin8nm")
DEMO_CSV = "demo_breast_cancer.csv"

# Value-shaped provider keys. Deliberately NOT the loose pre-commit pattern
# (api[_-]?key), which every config module matches by name.
SECRET_VALUE_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9]{16,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{16,}"),
)

_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".pytest_cache", "*.sqlite", ".env", ".env.*"
)


def assemble(staging: Path) -> Path:
    """Build the Space repo tree under ``staging`` and return it."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for f in ("Dockerfile", "README.md", "start.sh"):
        shutil.copy2(DEPLOY_SRC / f, staging / f)
    shutil.copy2(ROOT / "requirements.txt", staging / "requirements.txt")

    shutil.copytree(ROOT / "crewml", staging / "crewml", ignore=_IGNORE)
    shutil.copytree(ROOT / "scripts", staging / "scripts", ignore=_IGNORE)
    shutil.copytree(ROOT / "results", staging / "results", ignore=_IGNORE)

    data_dst = staging / "data"
    data_dst.mkdir()
    for name in BENCHMARK_DATASETS:
        shutil.copytree(ROOT / "data" / name, data_dst / name)
    shutil.copy2(ROOT / "data" / DEMO_CSV, data_dst / DEMO_CSV)
    return staging


def scan_for_secrets(staging: Path) -> list[str]:
    """Return ``path:pattern`` hits for key-VALUE shapes in text files."""
    hits: list[str] = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # parquet/gif/binary — key values are text artifacts
        for pat in SECRET_VALUE_PATTERNS:
            if pat.search(text):
                hits.append(f"{path.relative_to(staging)}:{pat.pattern}")
    hits.extend(
        f"{p.relative_to(staging)}:dotenv-file"
        for p in staging.rglob(".env*")
        if p.name != ".env.example"
    )
    return hits


def deploy(staging: Path, *, set_secret: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        SPACE_ID, repo_type="space", space_sdk="docker", exist_ok=True
    )
    if set_secret:
        # Read the key at deploy time from the local .env (config already
        # loaded it into the environment) — the value goes straight to the
        # Space's secret store and never touches the staging tree.
        from crewml.config import GROQ_API_KEY

        if not GROQ_API_KEY:
            raise SystemExit("--set-secret: no GROQ_API_KEY in local env/.env")
        api.add_space_secret(SPACE_ID, "GROQ_API_KEY", GROQ_API_KEY)
        print("Space secret GROQ_API_KEY set (value from local .env).")
    api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=str(staging),
        commit_message="Day 30: deploy CrewML Space (single-container API + dashboard)",
        delete_patterns=["*"],  # the staging tree IS the Space repo, exactly
    )
    print(f"Uploaded. {SPACE_URL}")


def wait_until_running(timeout_s: int) -> str:
    """Poll the Space build until RUNNING or a terminal error state."""
    from huggingface_hub import HfApi

    api = HfApi()
    deadline = time.monotonic() + timeout_s
    stage = "UNKNOWN"
    while time.monotonic() < deadline:
        stage = api.get_space_runtime(SPACE_ID).stage
        print(f"  space stage: {stage}")
        if stage == "RUNNING":
            return stage
        if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            raise SystemExit(f"Space entered {stage} — see {SPACE_URL}")
        time.sleep(20)
    raise SystemExit(f"Timed out after {timeout_s}s (last stage: {stage})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy the CrewML HF Space.")
    ap.add_argument("--staging-dir", type=Path, default=None,
                    help="assemble here instead of a temp dir (kept afterwards)")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble + secret-scan only; no upload")
    ap.add_argument("--set-secret", action="store_true",
                    help="also set the GROQ_API_KEY Space secret from local .env")
    ap.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                    help="after upload, poll until the Space is RUNNING")
    args = ap.parse_args()

    tmp = None
    if args.staging_dir is None:
        tmp = tempfile.mkdtemp(prefix="crewml-space-")
        staging = Path(tmp) / "space"
    else:
        staging = args.staging_dir

    try:
        assemble(staging)
        n_files = sum(1 for p in staging.rglob("*") if p.is_file())
        print(f"Assembled {n_files} files at {staging}")

        hits = scan_for_secrets(staging)
        if hits:
            for h in hits:
                print(f"  SECRET-SCAN HIT: {h}", file=sys.stderr)
            raise SystemExit("Refusing to deploy: secret scan failed.")
        print("Secret scan clean.")

        if args.dry_run:
            print("Dry run — stopping before upload.")
            return 0

        deploy(staging, set_secret=args.set_secret)
        if args.wait:
            wait_until_running(args.wait)
            print(f"Space is RUNNING: {SPACE_URL}")
        return 0
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
