"""Day 23 guard (landed Day 20): every committed artifact must match its generator.

A Day-20 audit of Days 1-20 found four committed files quietly disagreeing with the
data they claimed to present — fixture charts shipped as the real board, a table
saying "2/2" beside an 18/18 JSON, a chart captioned "no live-provider arm ran" over
the live arm, and a model card describing an executor that had since been hardened.
Every one was the same defect: an artifact drifting from its own generator. No amount
of care catches that class; re-deriving and comparing does.

These tests re-render each artifact from its committed source and demand an exact
match. When one fails the fix is almost always "regenerate and commit", never "relax
the assertion" — a loosened check here is worth nothing, since the whole point is that
the artifact and its data cannot silently disagree.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from crewml.artifact_registry import NOT_REDERIVABLE, derived_artifacts
from crewml.config import RESULTS_DIR

ARTIFACTS = derived_artifacts()
MARKDOWN = [a for a in ARTIFACTS if a.kind == "markdown"]
CHARTS = [a for a in ARTIFACTS if a.kind == "chart"]


def _load(artifact):
    return json.loads(artifact.source.read_text(encoding="utf-8"))


@pytest.mark.parametrize("artifact", MARKDOWN, ids=[a.name for a in MARKDOWN])
def test_committed_markdown_matches_its_generator(artifact):
    assert artifact.source.is_file(), f"source missing: {artifact.source}"
    assert artifact.path.is_file(), f"artifact missing: {artifact.path}"

    regenerated = artifact.render(_load(artifact))
    committed = artifact.path.read_text(encoding="utf-8")
    if regenerated == committed:
        return

    # Give a diagnosis, not just "not equal" — the failure that motivated this test
    # was a headline number differing by a factor of nine.
    import difflib

    diff = "\n".join(
        list(difflib.unified_diff(
            committed.splitlines(), regenerated.splitlines(),
            fromfile=f"committed/{artifact.name}", tofile="regenerated", lineterm="",
        ))[:40]
    )
    pytest.fail(
        f"{artifact.name} has drifted from {artifact.source.name} "
        f"({artifact.why}).\nRegenerate and commit it.\n\n{diff}"
    )


@pytest.mark.parametrize("artifact", CHARTS, ids=[a.name for a in CHARTS])
def test_committed_chart_matches_its_generator(artifact, tmp_path):
    assert artifact.source.is_file(), f"source missing: {artifact.source}"
    assert artifact.path.is_file(), f"artifact missing: {artifact.path}"

    fresh = tmp_path / artifact.path.name
    artifact.render(_load(artifact), fresh)

    committed_bytes = artifact.path.read_bytes()
    fresh_bytes = fresh.read_bytes()
    if committed_bytes == fresh_bytes:
        return

    pytest.fail(
        f"{artifact.name} does not match a fresh render of {artifact.source.name} "
        f"({artifact.why}).\n"
        f"  committed: {len(committed_bytes)} bytes, sha256 "
        f"{hashlib.sha256(committed_bytes).hexdigest()[:16]}\n"
        f"  regenerated: {len(fresh_bytes)} bytes, sha256 "
        f"{hashlib.sha256(fresh_bytes).hexdigest()[:16]}\n"
        "Regenerate and commit it. (If a matplotlib upgrade changed the encoder, "
        "regenerate — do not weaken this comparison.)"
    )


def test_chart_renders_are_byte_deterministic(tmp_path):
    """The premise the chart test rests on: same data in, same bytes out.

    If this ever fails, byte comparison above is not a valid check and the registry
    needs a different comparison strategy — so this failing must be loud, not a
    mysterious flake in the other tests.
    """
    artifact = CHARTS[0]
    data = _load(artifact)
    hashes = []
    for i in range(2):
        p = tmp_path / f"render{i}.png"
        artifact.render(data, p)
        hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
    assert hashes[0] == hashes[1], (
        "matplotlib PNG output is no longer deterministic on this install; the "
        "byte-comparison strategy in this module is invalid until revisited"
    )


def test_every_committed_generated_file_is_either_checked_or_excused():
    """No generated artifact may sit in results/ unchecked and unexplained.

    This is what stops the registry rotting: add a new board or chart and forget to
    register it, and this test fails until it is either covered or explicitly excused
    in NOT_REDERIVABLE with a reason.
    """
    checked = {a.name for a in ARTIFACTS}
    excused = set(NOT_REDERIVABLE)

    # Everything committed under results/ that is a rendered view of data.
    candidates = {
        str(p.relative_to(RESULTS_DIR)).replace("\\", "/")
        for p in RESULTS_DIR.rglob("*")
        if p.is_file() and p.suffix in {".md", ".png"}
    }
    unaccounted = candidates - checked - excused
    assert not unaccounted, (
        "these generated files under results/ are neither reproducibility-checked "
        f"nor listed in NOT_REDERIVABLE with a reason: {sorted(unaccounted)}"
    )


def test_model_card_boilerplate_tracks_the_reporter_template():
    """The sample card cannot be re-derived, so pin the part that went stale.

    `sample_model_card.md` told readers the executor was a process-isolation sandbox
    and self-repair was future work for two days after both shipped. The card's
    numbers are a snapshot, but its *claims about the system* must not outlive the
    system.
    """
    card = (RESULTS_DIR / "sample_model_card.md").read_text(encoding="utf-8")

    # Claims that were true once and are now false.
    for dead in ("process-isolation** sandbox, not yet a security sandbox",
                 "hardening is Phase 4 / Day 19",
                 "self-repair is Day 20"):
        assert dead not in card, f"stale claim still published in the model card: {dead!r}"

    # The current, true description of what the executor actually enforces.
    assert "SandboxPolicy" in card
    assert "self-repair" in card.lower()
