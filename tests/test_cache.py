"""Day 25 guards: the node cache returns the same answer, or no answer.

The cache's one honesty obligation is outcome-invisibility: a warm run must be
byte-identical to a cold run wherever results are concerned, and every hit must
be attributable (annotated provenance + a cache event in the state). These
tests pin the key discipline (content-addressed, code- and data-sensitive),
the fail-open behaviour, the never-memoise-an-outage rule, and the node wiring
(first-pass-only for the Planner, bypass on Critic re-entry).

All node-level tests disable the LLM narrative toggles: the cache's job here is
the deterministic core; live-narrative reuse is pinned by the llm-identity key
tests without touching a provider.
"""
from __future__ import annotations

import json

import pytest

from crewml import cache
from crewml.crew import nodes
from crewml.crew.planner import build_plan
from crewml.manifest import canonical_result


@pytest.fixture()
def cache_env(tmp_path, monkeypatch):
    """A private cache dir, cache on, LLM narratives off (deterministic core).

    Also forces the FILE backend: a developer shell exporting
    CREWML_REDIS_URL (e.g. at the compose stack) must not leak a live Redis —
    and its client memo — into tests that assert on cache files.
    """
    monkeypatch.setenv("CREWML_NODE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CREWML_NODE_CACHE", "1")
    monkeypatch.setenv("CREWML_PROFILER_LLM", "0")
    monkeypatch.setenv("CREWML_PLANNER_LLM", "0")
    monkeypatch.delenv("CREWML_REDIS_URL", raising=False)
    monkeypatch.setattr(cache, "_redis", None)
    return tmp_path


# --- Keys: content-addressed, sensitive to exactly the named inputs ----------

def test_cache_key_is_stable_and_pin_sensitive():
    pins = {"a": 1, "b": {"c": [1, 2]}}
    assert cache.cache_key("profile", pins) == cache.cache_key("profile", dict(pins))
    assert cache.cache_key("profile", pins) != cache.cache_key("plan", pins)
    assert cache.cache_key("profile", pins) != cache.cache_key("profile", {"a": 2, "b": pins["b"]})


def test_content_hash_ignores_narrative_and_annotation():
    core = {"n_rows": 10, "features": {"x": 1}}
    annotated = dict(core, llm_narrative={"text": "prose"},
                     cache={"hit": True, "key": "k"})
    assert cache.content_hash(core) == cache.content_hash(annotated)
    assert cache.content_hash(core) != cache.content_hash(dict(core, n_rows=11))


def test_source_sha256_varies_by_module():
    from crewml import leakage, telemetry
    assert cache.source_sha256(leakage) != cache.source_sha256(telemetry)
    assert cache.source_sha256(leakage) == cache.source_sha256(leakage)


# --- Store: roundtrip, fail-open, kill switch --------------------------------

def test_lookup_store_roundtrip_annotates_hits(cache_env):
    pins = {"x": 1}
    value = {"n_rows": 5, "assessment": {"flags": []}}
    assert cache.lookup("profile", pins) is None
    assert cache.store("profile", pins, value) is True

    hit = cache.lookup("profile", pins)
    assert hit is not None
    assert hit["cache"]["hit"] is True
    assert hit["cache"]["key"] == cache.cache_key("profile", pins)
    assert {k: v for k, v in hit.items() if k != "cache"} == value
    # The hit is a copy — mutating it must not poison the stored entry.
    hit["n_rows"] = 999
    assert cache.lookup("profile", pins)["n_rows"] == 5


def test_hit_annotation_is_never_persisted(cache_env):
    pins = {"x": 2}
    cache.store("profile", pins, {"v": 1, "cache": {"hit": True, "key": "stale"}})
    entry = json.loads(next(cache_env.glob("profile-*.json")).read_text(encoding="utf-8"))
    assert "cache" not in entry["value"]


def test_disabled_cache_is_inert(cache_env, monkeypatch):
    monkeypatch.setenv("CREWML_NODE_CACHE", "0")
    assert cache.store("profile", {"x": 1}, {"v": 1}) is False
    assert cache.lookup("profile", {"x": 1}) is None
    assert list(cache_env.iterdir()) == []


def test_corrupt_or_mismatched_entry_misses(cache_env):
    pins = {"x": 3}
    cache.store("profile", pins, {"v": 1})
    path = next(cache_env.glob("profile-*.json"))

    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["key"] = "0" * 64  # stale/foreign entry under the right filename
    path.write_text(json.dumps(entry), encoding="utf-8")
    assert cache.lookup("profile", pins) is None

    path.write_text("{not json", encoding="utf-8")
    assert cache.lookup("profile", pins) is None  # fail-open, never raises


# --- The never-memoise-an-outage rule ---------------------------------------

def test_degraded_live_narrative_is_not_cacheable():
    live = {"enabled": True, "mock_mode": False, "provider": "groq", "model": "m"}
    outage = {"llm_narrative": {"source": "unavailable", "reason": "boom"}}
    healthy = {"llm_narrative": {"source": "groq", "text": "briefing"}}
    assert cache.value_cacheable(outage, live) is False
    assert cache.value_cacheable(healthy, live) is True
    # Mock/disabled runs record `unavailable` as their steady state — cacheable.
    assert cache.value_cacheable(outage, dict(live, mock_mode=True)) is True
    assert cache.value_cacheable(outage, dict(live, enabled=False)) is True


# --- Node wiring: profiler ---------------------------------------------------

def test_profiler_node_cold_then_warm(cache_env, monkeypatch):
    calls = {"n": 0}
    real = nodes.run_profiler

    def counting(key, **kw):
        calls["n"] += 1
        return real(key, **kw)

    monkeypatch.setattr(nodes, "run_profiler", counting)
    state = {"dataset_key": "credit-g"}

    cold = nodes.profiler(state)
    assert calls["n"] == 1
    [event] = cold["cache_events"]
    assert event == {"node": "profiler", "kind": "profile", "hit": False,
                     "stored": True, "key": event["key"]}

    warm = nodes.profiler(state)
    assert calls["n"] == 1  # served from cache — the node never recomputed
    [event2] = warm["cache_events"]
    assert event2["hit"] is True

    cold_p, warm_p = cold["profile"], dict(warm["profile"])
    provenance = warm_p.pop("cache")
    assert provenance["hit"] is True
    assert warm_p == cold_p  # byte-identical answer, annotation aside


def test_profiler_bypasses_on_unsealed_dataset(cache_env, monkeypatch):
    monkeypatch.setattr(nodes, "run_profiler", lambda key, **kw: {"n_rows": 1})
    out = nodes.profiler({"dataset_key": "no-such-dataset"})
    [event] = out["cache_events"]
    assert event["bypass"] == "no_dataset_seals"
    assert out["profile"] == {"n_rows": 1}
    assert list(cache_env.iterdir()) == []  # nothing stored for the unpinnable


# --- Node wiring: planner (first pass only) ----------------------------------

def _first_pass_state(cache_env):
    cold = nodes.profiler({"dataset_key": "credit-g"})
    return {"dataset_key": "credit-g", "profile": cold["profile"],
            "critiques": [], "iteration": 0}


def test_planner_node_cold_then_warm_and_hit_agnostic_to_profile_annotation(cache_env):
    state = _first_pass_state(cache_env)
    cold = nodes.planner(state)
    assert cold["cache_events"][0]["hit"] is False

    # Second run's profile came from a profiler HIT (carries the annotation);
    # the plan key must strip it and still hit.
    warm_profile = nodes.profiler({"dataset_key": "credit-g"})["profile"]
    assert "cache" in warm_profile
    warm = nodes.planner(dict(state, profile=warm_profile))
    assert warm["cache_events"][0]["hit"] is True

    warm_plan = dict(warm["plan"])
    warm_plan.pop("cache")
    assert warm_plan == cold["plan"]


def test_planner_bypasses_critic_reentry(cache_env):
    state = _first_pass_state(cache_env)
    critique = {"decision": "iterate", "findings": ["overfit: gap too large"]}
    out = nodes.planner(dict(state, critiques=[critique], iteration=1))
    [event] = out["cache_events"]
    assert event["bypass"] == "critique_reentry"
    assert out["plan"]["addressed_critique"] == critique
    assert not list(cache_env.glob("plan-*.json"))  # re-entry plans never stored

    # And the loop stays live on a later first-pass-shaped call: the stored
    # first-pass plan must not leak into an iteration>0 build.
    out2 = nodes.planner(dict(state, critiques=[critique], iteration=2))
    assert out2["plan"]["planning_for_iteration"] == 2


# --- Outcome-invisibility ----------------------------------------------------

def test_cache_events_never_reach_the_result_fingerprint(cache_env):
    state = _first_pass_state(cache_env)
    final_shaped = dict(state, cache_events=[{"node": "profiler", "hit": True}],
                        trace=["profiler", "planner"], critiques=[])
    assert "cache_events" not in json.dumps(canonical_result(final_shaped))


def test_warm_profile_builds_identical_plan(cache_env):
    cold_profile = nodes.profiler({"dataset_key": "credit-g"})["profile"]
    warm_profile = nodes.profiler({"dataset_key": "credit-g"})["profile"]
    assert build_plan(cold_profile) == build_plan(warm_profile)


# --- Day 27: Redis backend — same schema, same honesty, fail-open twice ------

class _FakeRedis:
    """The two methods the backend uses, over a plain dict. decode_responses=True."""

    def __init__(self):
        self.data: dict[str, str] = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


class _DeadRedis:
    def get(self, key):  # pragma: no cover - trivial
        raise ConnectionError("redis down")

    def set(self, key, value):
        raise ConnectionError("redis down")


@pytest.fixture()
def redis_reset(monkeypatch):
    """Fresh client memo per test — a prior test's dead-memo must not leak."""
    monkeypatch.setattr(cache, "_redis", None)


def test_unconfigured_redis_uses_file_backend(cache_env, redis_reset, monkeypatch):
    monkeypatch.delenv("CREWML_REDIS_URL", raising=False)
    assert cache._redis_client() is None
    assert cache.store("profile", {"p": 1}, {"v": 1})
    assert list(cache_env.glob("profile-*.json"))  # entry landed on disk


def test_redis_roundtrip_annotates_hits_and_skips_files(cache_env, redis_reset,
                                                        monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "_redis_client", lambda: fake)
    pins = {"train_sha256": "abc", "seed": 42}
    assert cache.store("profile", pins, {"n_rows": 10, cache.CACHE_META_KEY: {"hit": True}})
    assert not list(cache_env.glob("profile-*.json"))  # redis replaced files
    [(key, raw)] = fake.data.items()
    assert key == cache._redis_key("profile", cache.cache_key("profile", pins))
    assert cache.CACHE_META_KEY not in json.loads(raw)["value"]  # annotation stripped
    hit = cache.lookup("profile", pins)
    assert hit["n_rows"] == 10
    assert hit[cache.CACHE_META_KEY]["hit"] is True


def test_redis_full_key_check_still_applies(cache_env, redis_reset, monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "_redis_client", lambda: fake)
    pins = {"p": 1}
    key = cache.cache_key("profile", pins)
    fake.data[cache._redis_key("profile", key)] = json.dumps(
        {"kind": "profile", "key": "not-the-key", "value": {"v": 1}})
    assert cache.lookup("profile", pins) is None  # corrupt entry misses, never lies


def test_dead_redis_falls_open_to_file_backend(cache_env, redis_reset, monkeypatch):
    monkeypatch.setenv("CREWML_REDIS_URL", "redis://localhost:1")  # configured...
    monkeypatch.setattr(cache, "_redis", _DeadRedis())  # ...but the server died
    assert cache.store("profile", {"p": 2}, {"v": 2})  # True: file took it
    assert list(cache_env.glob("profile-*.json"))
    assert cache._redis is cache._REDIS_DEAD  # memoised: no repeat timeouts
    assert cache.lookup("profile", {"p": 2})["v"] == 2  # served from disk


def test_unreachable_redis_url_memoises_dead(cache_env, redis_reset, monkeypatch):
    # A configured-but-unconnectable URL: first call pays the 1s ping, then None.
    monkeypatch.setenv("CREWML_REDIS_URL", "redis://127.0.0.1:1/0")
    assert cache._redis_client() is None
    assert cache._redis is cache._REDIS_DEAD
    assert cache.store("profile", {"p": 3}, {"v": 3})  # file backend still works


def test_corrupt_redis_entry_is_a_miss_not_a_dead_server(cache_env, redis_reset,
                                                         monkeypatch):
    """One poisoned value must cost one key, not the whole shared cache.

    Audit finding (Day 27): decode failure was conflated with server failure
    and memoised _REDIS_DEAD — a single corrupt entry permanently downgraded
    the process to file caching. Corrupt ENTRY => per-key miss, server LIVE.
    """
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "_redis_client", lambda: fake)
    good_pins, bad_pins = {"p": "good"}, {"p": "bad"}
    assert cache.store("profile", good_pins, {"v": 1})
    fake.data[cache._redis_key("profile", cache.cache_key("profile", bad_pins))] = \
        "{not json"
    assert cache.lookup("profile", bad_pins) is None      # corrupt -> miss
    assert cache._redis is not cache._REDIS_DEAD          # server NOT declared dead
    assert cache.lookup("profile", good_pins)["v"] == 1   # still served from redis
    assert not list(cache_env.glob("profile-*.json"))     # and never from files
