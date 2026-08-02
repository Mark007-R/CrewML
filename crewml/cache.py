"""Content-addressed node cache — profiles and plans keyed by what made them (Day 25).

The Profiler and (first-pass) Planner are deterministic functions of things we
can name exactly: the train split's bytes, the node's source code, its schema
version, the seed, and — for the advisory narrative — the LLM identity. Same
inputs, same output; recomputing them on every ``/run`` for the same dataset
re-spends the leakage screen's CV passes and two LLM narrative calls to produce
a byte-identical answer. This module memoises exactly that and nothing more.

What is deliberately NOT cached:

* **Critic-loop re-entries.** A plan built against a critique depends on the
  whole run history; the planner node bypasses the cache whenever a critique
  exists (recorded as a ``bypass`` cache event, so telemetry shows the loop was
  live, not cache-starved).
* **Training / ensembling.** Their cost is compute on data, not recomputable
  bookkeeping — and their artifacts (fitted models) live outside the state.
* **Degraded narratives.** A profile/plan whose LLM narrative came back
  ``unavailable`` while a live provider was *expected* is a transient failure,
  not a fact about the inputs — storing it would replay the outage forever.
  (:func:`value_cacheable` is that rule; mock/disabled narratives are the
  expected steady state and cache fine.)

Honesty invariant: a cache hit must be **outcome-invisible** — the Day-23
result fingerprint of a warm run equals the cold run's, because the cached
value is byte-identical to what the node would recompute. The only trace a hit
leaves is provenance: the returned copy carries a ``cache`` annotation and the
run's telemetry counts the hit. Keys never contain data, only hashes; the
cache directory lives under the git-ignored ``artifacts/``.

Every public function is fail-open: a corrupt entry, an unwritable directory,
or a missing dataset seal degrades to "no cache", never to a crashed node.

Backends (Day 27): entries live in per-process JSON files under the
git-ignored ``artifacts/`` by default; setting ``CREWML_REDIS_URL`` switches
lookup/store to a shared Redis with the identical entry schema, so hits
survive container rebuilds and multiple API containers share one cache
(docker-compose wires ``redis://redis:6379/0``). Redis inherits the fail-open
rule twice over: an unreachable server degrades to the file backend — never a
crashed node — and is remembered dead for the rest of the process so each call
doesn't re-pay a connection timeout.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from crewml import config
from crewml.manifest import canonical_json

CACHE_SCHEMA_VERSION = 1

# Annotation key attached to values returned from a hit — provenance, not data.
# Stripped by content_hash() so a hit-derived value hashes like the original.
CACHE_META_KEY = "cache"

# Value keys that legitimately differ between two honest computations of the
# same pinned inputs (narrative prose + our own annotation) — excluded from
# content hashing so downstream keys agree across cold and warm runs.
VOLATILE_VALUE_KEYS = (CACHE_META_KEY, "llm_narrative")


def enabled() -> bool:
    """Cache master switch — ``CREWML_NODE_CACHE`` (default on; ``0`` disables).

    Read at call time (not import) so tests and the API's env plumbing can
    toggle it per run; being a ``CREWML_`` variable it lands in the Day-23
    manifest's environment capture automatically.
    """
    return os.getenv("CREWML_NODE_CACHE", "1").lower() not in ("0", "false", "off")


def cache_dir() -> Path:
    """Where entries live — git-ignored artifacts/, overridable for tests."""
    return Path(os.getenv("CREWML_NODE_CACHE_DIR", str(config.ARTIFACTS_DIR / "cache")))


# --- Keys ---------------------------------------------------------------------

def cache_key(kind: str, pins: dict[str, Any]) -> str:
    """SHA-256 over the canonical pins — the entry's full identity."""
    payload = {"cache_schema": CACHE_SCHEMA_VERSION, "kind": kind, "pins": pins}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def content_hash(value: dict[str, Any],
                 exclude: tuple[str, ...] = VOLATILE_VALUE_KEYS) -> str:
    """Hash a node output minus its volatile keys (narrative, cache annotation)."""
    core = {k: v for k, v in value.items() if k not in exclude}
    return hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()


def source_sha256(*modules: ModuleType) -> str:
    """One hash over the given modules' source files — code changes invalidate.

    Falls back to the module's version-less name if a source file is unreadable
    (frozen environments); the key then still varies by module identity.
    """
    h = hashlib.sha256()
    for mod in modules:
        try:
            h.update(Path(mod.__file__).read_bytes())
        except Exception:
            h.update(mod.__name__.encode("utf-8"))
    return h.hexdigest()


def llm_pins(env_toggle: str) -> dict[str, Any]:
    """The LLM identity a narrative depends on: toggle, mock-mode, provider, model."""
    return {
        "enabled": os.getenv(env_toggle, "1") != "0",
        "mock_mode": config.is_mock_mode(),
        "provider": config.LLM_PROVIDER,
        "model": (config.GROQ_MODEL if config.LLM_PROVIDER == "groq"
                  else config.ANTHROPIC_MODEL),
    }


def value_cacheable(value: dict[str, Any], llm: dict[str, Any]) -> bool:
    """False when a live narrative was expected but came back ``unavailable``.

    A transient provider failure must not be memoised as if it were the answer;
    mock/disabled runs record ``unavailable`` as their steady state and cache fine.
    """
    if not llm.get("enabled") or llm.get("mock_mode"):
        return True
    narrative = value.get("llm_narrative") or {}
    return narrative.get("source") != "unavailable"


# --- Redis backend (Day 27) ---------------------------------------------------

_REDIS_DEAD = object()  # module-level memo: "configured but unreachable"
_redis: Any = None      # None = not yet initialised; _REDIS_DEAD = gave up


def redis_url() -> str:
    """Shared-cache switch — empty (default) keeps the file backend."""
    return os.getenv("CREWML_REDIS_URL", "").strip()


def _redis_key(kind: str, key: str) -> str:
    return f"crewml:cache:{kind}:{key}"


def _redis_client() -> Optional[Any]:
    """A live Redis client, or None (unconfigured, import failed, or dead).

    The client is built once per process with 1-second socket timeouts and
    ping-verified; any failure memoises _REDIS_DEAD so later lookups skip
    straight to the file backend instead of re-paying the timeout. Tests
    monkeypatch this function to inject fakes.
    """
    global _redis
    url = redis_url()
    if not url:
        return None
    if _redis is _REDIS_DEAD:
        return None
    if _redis is not None:
        return _redis
    try:
        import redis as _redis_mod

        client = _redis_mod.Redis.from_url(
            url, socket_connect_timeout=1, socket_timeout=1,
            decode_responses=True,
        )
        client.ping()
        _redis = client
        return _redis
    except Exception:
        _redis = _REDIS_DEAD
        return None


def _mark_redis_dead() -> None:
    global _redis
    _redis = _REDIS_DEAD


# --- Store --------------------------------------------------------------------

def _entry_path(kind: str, key: str) -> Path:
    return cache_dir() / f"{kind}-{key[:16]}.json"


def _load_entry(kind: str, key: str) -> Optional[dict[str, Any]]:
    """Fetch the raw entry dict from Redis (if configured+alive) else file."""
    client = _redis_client()
    if client is not None:
        try:
            raw = client.get(_redis_key(kind, key))
        except Exception:
            _mark_redis_dead()  # SERVER failure -> file backend is never worse
        else:
            if not raw:
                return None  # authoritative miss; file is not a second chance
            try:
                return json.loads(raw)
            except Exception:
                # Corrupt ENTRY, live server: a per-key miss, exactly like a
                # corrupt file — conflating it with a dead server would let one
                # poisoned value disable the shared cache for the whole process.
                return None
    try:
        return json.loads(_entry_path(kind, key).read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_entry(kind: str, key: str, entry: dict[str, Any]) -> bool:
    """Persist the entry to Redis (if configured+alive) else file (atomic)."""
    client = _redis_client()
    if client is not None:
        try:
            client.set(_redis_key(kind, key), json.dumps(entry, default=str))
            return True
        except Exception:
            _mark_redis_dead()
    path = _entry_path(kind, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def lookup(kind: str, pins: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return an annotated deep copy of the cached value, or None. Never raises."""
    if not enabled():
        return None
    key = cache_key(kind, pins)
    entry = _load_entry(kind, key)
    if entry is None:
        return None
    try:
        # Full-key check: the filename carries only a prefix; a (vanishingly
        # unlikely) prefix collision or a stale/corrupt entry must miss, not lie.
        if entry.get("key") != key or entry.get("kind") != kind:
            return None
        value = copy.deepcopy(entry["value"])
    except Exception:
        return None
    if isinstance(value, dict):
        value[CACHE_META_KEY] = {
            "hit": True, "kind": kind, "key": key,
            "created_at": entry.get("created_at"),
        }
    return value


def store(kind: str, pins: dict[str, Any], value: dict[str, Any]) -> bool:
    """Persist one entry; True on success. Never raises."""
    if not enabled():
        return False
    key = cache_key(kind, pins)
    entry = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "kind": kind,
        "key": key,
        "pins": pins,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Store the value WITHOUT any hit annotation a caller may have added.
        "value": {k: v for k, v in value.items() if k != CACHE_META_KEY},
    }
    return _save_entry(kind, key, entry)
