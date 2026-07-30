"""Thin LLM provider abstraction — Groq (default) with an offline mock fallback.

Every LLM call in CrewML goes through :func:`chat`, which returns an
:class:`LLMResult` carrying the completion text *and* token accounting so that
run budgets (``CREWML_RUN_TOKEN_BUDGET``) and the Day 16 provider study can be
measured uniformly. When no API key is configured the pipeline is in **mock
mode** (:func:`crewml.config.is_mock_mode`) and :func:`chat` raises
:class:`MockModeError` — callers are expected to check ``is_mock_mode()`` first
and take a deterministic offline path so no network is ever required to run.

The honesty rule (EVAL_PROTOCOL.md §5): a result produced without a live LLM is
**mock** and must be labelled as such; it is never the headline number.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from crewml import budget, config


class MockModeError(RuntimeError):
    """Raised when a live LLM call is attempted while in mock mode."""


@dataclass(frozen=True)
class LLMResult:
    """One completion plus the accounting the rest of the system needs."""

    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    is_mock: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _chat_groq(system: str, user: str, *, temperature: float, max_tokens: int) -> LLMResult:
    """Call Groq's chat-completions endpoint. Imported lazily so mock runs need no SDK."""
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        # Seed the sampler for reproducibility (best-effort on Groq) so a given
        # (prompt, temperature) re-runs to the same code — the project is
        # seed-locked everywhere else (EVAL_PROTOCOL §3.1).
        seed=config.SEED,
    )
    usage = resp.usage
    return LLMResult(
        text=resp.choices[0].message.content or "",
        provider="groq",
        model=config.GROQ_MODEL,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0)),
        completion_tokens=int(getattr(usage, "completion_tokens", 0)),
        is_mock=False,
    )


def _chat_anthropic(system: str, user: str, *, temperature: float, max_tokens: int) -> LLMResult:
    """Call Anthropic's Messages API (optional provider for the Day 16 study)."""
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    return LLMResult(
        text=text,
        provider="anthropic",
        model=config.ANTHROPIC_MODEL,
        prompt_tokens=int(resp.usage.input_tokens),
        completion_tokens=int(resp.usage.output_tokens),
        is_mock=False,
    )


def chat(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    agent: str = "unspecified",
) -> LLMResult:
    """Send a system+user prompt to the configured provider and return the result.

    Raises :class:`MockModeError` in mock mode — callers must branch on
    :func:`crewml.config.is_mock_mode` before calling and provide an offline path.

    When a run budget is active (:mod:`crewml.budget`, Day 21) the call is gated
    on it *before* the network is touched — an exhausted budget raises
    :class:`crewml.budget.BudgetExhaustedError`, which every caller already
    degrades on gracefully — and charged to it after, under the caller's
    ``agent`` label so the ledger can itemise the run's cost per agent. With no
    active budget (unit tests, ad-hoc probes) behaviour is unchanged.
    """
    if config.is_mock_mode():
        raise MockModeError(
            "No LLM key configured (mock mode). Call is_mock_mode() first and take "
            "the offline path; mock output is never reported as a real result."
        )
    run_budget = budget.active()
    if run_budget is not None:
        run_budget.enforce(agent=agent)  # raises BudgetExhaustedError when spent
    started = time.monotonic()
    if config.LLM_PROVIDER == "groq":
        result = _chat_groq(system, user, temperature=temperature, max_tokens=max_tokens)
    elif config.LLM_PROVIDER in ("anthropic", "claude"):
        result = _chat_anthropic(system, user, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"unknown LLM provider {config.LLM_PROVIDER!r}")
    if run_budget is not None:
        run_budget.charge(
            agent=agent,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_s=time.monotonic() - started,  # Day 25 telemetry
        )
    return result


_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(text: str) -> str:
    """Pull the Python source out of an LLM reply.

    Prefers the first fenced ```python block; falls back to the largest fenced
    block; finally returns the raw text if the model emitted bare code. Raising
    is left to the caller once it tries to compile the result.
    """
    blocks = _CODE_FENCE.findall(text)
    if blocks:
        # Prefer the longest block — models sometimes emit a tiny example first.
        return max(blocks, key=len).strip()
    return text.strip()
