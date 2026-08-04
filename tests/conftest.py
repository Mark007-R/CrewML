"""Suite-wide fixtures and invariants (Day 28).

Two things every test in the suite gets for free:

* **Registry isolation.** ``crewml.datasets.REGISTRY`` is module-level state
  that upload restoration mutates in place (by design — ``/run`` looks every
  dataset up there). A test that registers an upload and forgets to clean up
  would silently widen "the suite" for every test that runs after it, so the
  snapshot below makes leakage impossible rather than merely impolite.

* **A benchmark-suite sanity gate.** The locked five-dataset registry is the
  ground truth almost every test leans on; if it has been tampered with, fail
  loudly once at session start instead of 500 times downstream.
"""
from __future__ import annotations

import pytest

from crewml.datasets import BENCHMARK_KEYS, REGISTRY


def pytest_sessionstart(session):
    missing = [k for k in BENCHMARK_KEYS if k not in REGISTRY]
    assert not missing, (
        f"benchmark datasets missing from REGISTRY: {missing} — "
        f"the locked suite has been tampered with"
    )


@pytest.fixture(autouse=True)
def _registry_isolation():
    """Restore REGISTRY to its pre-test membership after every test."""
    before = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(before)
