"""Day 30 guards: the Hugging Face Space ships exactly what it should.

The Space repo is assembled by ``scripts/deploy_hf_space.py`` rather than
pushed from the working tree, so these tests pin the assembly contract:

  * the Space-card front matter says ``sdk: docker`` / ``app_port: 7860`` and
    the Dockerfile exposes that port and runs both processes via start.sh;
  * the staging tree contains the code, the five sealed benchmark splits and
    the demo CSV — and does NOT contain dotenv files, host-local study
    fixtures (``probe_*``/``upload-*`` data), caches, or the run store;
  * the secret scan catches key-VALUE shapes (a planted ``gsk_…``) while
    tolerating legitimate variable NAMES like GROQ_API_KEY in code.

Everything runs against a temp staging dir — no network, no HF calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402

import deploy_hf_space as dhs  # noqa: E402


@pytest.fixture(scope="module")
def staging(tmp_path_factory) -> Path:
    return dhs.assemble(tmp_path_factory.mktemp("space") / "tree")


def _front_matter(readme: Path) -> dict[str, str]:
    lines = readme.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---", "Space card must start with YAML front matter"
    fm: dict[str, str] = {}
    for line in lines[1:]:
        if line == "---":
            return fm
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip()
    raise AssertionError("unterminated front matter")


# --- Space card + Dockerfile contract -------------------------------------

def test_card_declares_docker_sdk_on_port_7860(staging: Path) -> None:
    fm = _front_matter(staging / "README.md")
    assert fm["sdk"] == "docker"
    assert fm["app_port"] == "7860"


def test_dockerfile_exposes_7860_and_boots_start_sh(staging: Path) -> None:
    df = (staging / "Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 7860" in df
    assert 'CMD ["bash", "start.sh"]' in df
    # No baked key: the provider key must arrive as a Space secret at runtime,
    # so no ENV/ARG line may assign GROQ_API_KEY (comments may explain this).
    assert not any(
        "GROQ_API_KEY=" in line
        for line in df.splitlines()
        if not line.lstrip().startswith("#")
    )


def test_start_sh_runs_api_privately_and_dashboard_on_7860(staging: Path) -> None:
    sh = (staging / "start.sh").read_text(encoding="utf-8")
    assert "uvicorn crewml.api.app:app --host 127.0.0.1 --port 8000" in sh
    assert "--server.port 7860" in sh
    assert "--server.address 0.0.0.0" in sh


# --- Staging tree contents -------------------------------------------------

def test_staging_has_code_and_evidence(staging: Path) -> None:
    for required in (
        "requirements.txt",
        "crewml/api/app.py",
        "crewml/dashboard/app.py",
        "scripts/prepare_datasets.py",
        "results/dataset_manifest.json",
    ):
        assert (staging / required).is_file(), f"missing {required}"


def test_staging_data_is_exactly_benchmarks_plus_demo(staging: Path) -> None:
    entries = {p.name for p in (staging / "data").iterdir()}
    assert entries == set(dhs.BENCHMARK_DATASETS) | {dhs.DEMO_CSV}
    for name in dhs.BENCHMARK_DATASETS:
        for split in ("train.parquet", "holdout.parquet"):
            assert (staging / "data" / name / split).is_file()


def test_staging_excludes_local_only_files(staging: Path) -> None:
    assert not list(staging.rglob(".env*")), "dotenv files must never ship"
    assert not list(staging.rglob("__pycache__"))
    assert not list(staging.rglob("*.sqlite")), "run store is host-local"
    assert not list(staging.rglob("probe_*")), "study fixtures are host-local"


# --- Secret scan -----------------------------------------------------------

def test_secret_scan_clean_on_real_assembly(staging: Path) -> None:
    assert dhs.scan_for_secrets(staging) == []


def test_secret_scan_catches_planted_key_value(tmp_path: Path) -> None:
    (tmp_path / "oops.py").write_text(
        'KEY = "gsk_' + "a1B2c3D4e5F6g7H8i9J0" + '"', encoding="utf-8"
    )
    hits = dhs.scan_for_secrets(tmp_path)
    assert hits and "oops.py" in hits[0]


def test_secret_scan_tolerates_variable_names(tmp_path: Path) -> None:
    (tmp_path / "config.py").write_text(
        'GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")', encoding="utf-8"
    )
    assert dhs.scan_for_secrets(tmp_path) == []


def test_secret_scan_flags_stray_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GROQ_API_KEY=x", encoding="utf-8")
    assert any("dotenv-file" in h for h in dhs.scan_for_secrets(tmp_path))
