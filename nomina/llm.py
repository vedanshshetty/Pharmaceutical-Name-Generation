"""
NOMINA LLM proposer — semantic name generation via OpenRouter, free tier only.

Why free-tier needs its own module
----------------------------------
v1 hardcoded `google/gemini-pro`. That slug is stale, and more importantly the whole
approach is fragile: OpenRouter's free roster rotates constantly, models get delisted
without notice, and a hardcoded ID turns into a silent 404 that reads to the user as
"the LLM strategy produced nothing". A project that claims production readiness cannot
have a single hardcoded third-party string as a load-bearing component.

So model selection is resolved at runtime, in this order:

1. An explicit model the caller asked for.
2. `openrouter/free`, OpenRouter's own auto-router, which selects a live free model per
   request and keeps working across roster changes.
3. Live discovery: query `/api/v1/models`, keep those whose prompt AND completion price
   are both zero, and rank them by a preference list.
4. A committed static fallback list.

Every call walks the chain until one succeeds. The LLM is therefore *never* a hard
dependency: with no key, no network, or an empty roster, `propose()` returns an empty
list and the orchestrator proceeds on the free CPU proposers alone.

Rate limits are real (free tier is roughly 20 requests/minute and a low daily cap), so
the proposer is batched by construction: one call returns N candidates, and the
orchestrator only reaches for it when the free pool has failed to produce something
good enough.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# The auto-router. Introduced specifically so callers stop hardcoding rotating slugs.
AUTO_ROUTER = "openrouter/free"

# Ranked preferences applied to whatever discovery returns. Instruction-following and
# constraint adherence matter far more here than raw reasoning: the task is "emit N
# strings that all end in these five letters and contain nothing else".
PREFERRED_SUBSTRINGS: Sequence[str] = (
    "llama-4", "llama-3.3", "gpt-oss", "qwen", "nemotron", "gemma", "mistral",
    "deepseek", "phi", "command",
)

STATIC_FALLBACK_MODELS: Sequence[str] = (
    AUTO_ROUTER,
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "google/gemma-3-27b-it:free",
)


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Key lookup order: explicit argument, then the standard environment variables,
    then Colab secrets. Never prompts, never logs the value."""
    if explicit:
        return explicit
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "NOMINA_LLM_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    try:                                                   # pragma: no cover - Colab only
        from google.colab import userdata
        for name in ("OPENROUTER_API_KEY", "llmkey", "openrouter"):
            try:
                v = userdata.get(name)
                if v:
                    return v
            except Exception:                              # noqa: BLE001
                continue
    except ImportError:
        pass
    return None


def discover_free_models(api_key: Optional[str] = None, timeout: float = 15) -> List[str]:
    """Live roster of genuinely zero-cost models, best first.

    Filters on `pricing.prompt == 0 and pricing.completion == 0` rather than on the
    `:free` suffix, because the suffix is a naming convention and the price field is the
    actual contract.
    """
    try:
        from .data_layer import _http_get
        raw = _http_get(f"{OPENROUTER_BASE}/models", timeout=timeout)
        data = json.loads(raw).get("data", [])
    except Exception:                                      # noqa: BLE001
        return []

    free = []
    for m in data:
        pricing = m.get("pricing") or {}
        try:
            if float(pricing.get("prompt", 1)) == 0.0 and float(pricing.get("completion", 1)) == 0.0:
                free.append(m.get("id"))
        except (TypeError, ValueError):
            continue
    free = [m for m in free if m]

    def rank(mid: str) -> int:
        low = mid.lower()
        for i, frag in enumerate(PREFERRED_SUBSTRINGS):
            if frag in low:
                return i
        return len(PREFERRED_SUBSTRINGS)

    return sorted(free, key=rank)


@dataclass
class LLMConfig:
    model: Optional[str] = None            # None => resolve the chain
    max_tokens: int = 900
    temperature: float = 1.0
    timeout: float = 60.0
    max_attempts: int = 4                  # distinct models to try before giving up
    enable_discovery: bool = True
    cooldown_s: float = 3.0                # free tier is ~20 rpm; be a good citizen


@dataclass
class LLMResult:
    names: List[str] = field(default_factory=list)
    model_used: Optional[str] = None
    attempts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.names)


_NAME_RE = re.compile(r"[a-z]{3,20}")


def _parse_names(text: str, n: int, required_suffix: str = "") -> List[str]:
    """Strict extraction. Models add numbering, bullets, preambles and trailing
    commentary no matter how firmly the prompt forbids it, so output is parsed rather
    than trusted: one lowercase alphabetic token per line, honouring the stem."""
    out: List[str] = []
    seen = set()
    for line in (text or "").splitlines():
        line = line.strip().strip("-*•\t ").lower()
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        for tok in _NAME_RE.findall(line):
            if required_suffix and not tok.endswith(required_suffix):
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            break                                          # at most one name per line
    return out[:n]


def build_prompt(target_type: str, target_class: Optional[str], target_stem: Optional[str],
                 n: int, avoid_names: Sequence[str] = (),
                 rejected_this_run: Sequence[str] = ()) -> str:
    """The prompt carries THIS run's rejections, not just a static sample.

    In v1 the LLM received a fixed list of real sibling names and nothing else, while
    the statistical proposer received live rejection feedback. That asymmetry meant the
    one proposer capable of reasoning semantically about why a name failed was the one
    proposer never told that anything had failed.
    """
    if target_type == "generic":
        stem = (target_stem or "").lstrip("-")
        constraint = (
            f"Every candidate MUST end in the exact letters '{stem}', the INN/USAN stem "
            f"for this class. The letters before the stem are the 'fantasy prefix'.\n"
            f"The fantasy prefix MUST NOT be a recognisable fragment of any existing "
            f"drug (do not produce things like 'erythro{stem}' or 'amoxi{stem}', which "
            f"borrow another class's morpheme). It MUST NOT contain any other INN stem. "
            f"Aim for 2-3 syllables in the prefix, easy to say in English, Spanish, "
            f"Mandarin and Arabic."
        )
    else:
        constraint = (
            "This is a PROPRIETARY (brand) name. It MUST NOT end in, or contain, any "
            "INN/USAN stem, because a stem falsely implies pharmacological class. It "
            "must be 6-9 letters, 2-3 syllables, distinctive and easy to say. It must "
            "NOT imply safety, efficacy or superiority (no 'cure', 'best', 'safe', "
            "'pure', 'vita'), and must not carry an adverse meaning in a major language."
        )

    blocks = [
        f"You are a pharmaceutical nomenclature specialist naming a new "
        f"{target_type} product.",
        f"Pharmacological class: {target_class or '(unspecified)'}",
        constraint,
    ]
    if avoid_names:
        blocks.append("These real marketed names already exist. Anything look-alike or "
                      "sound-alike to them is rejected: " + ", ".join(list(avoid_names)[:20]) + ".")
    if rejected_this_run:
        blocks.append("These candidates were already generated and REJECTED by the "
                      "regulatory screen in this session. Do not repeat them or produce "
                      "near-variants: " + ", ".join(list(rejected_this_run)[:20]) + ".")
    blocks.append(
        f"Output exactly {n} candidates, one per line, lowercase a-z only, no numbering, "
        f"no bullets, no explanation, no blank lines. Nothing but the {n} names.")
    return "\n\n".join(blocks)


class OpenRouterProposer:
    """Batched LLM proposer that degrades to nothing rather than to an exception."""

    def __init__(self, config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        self.config = config or LLMConfig()
        self.api_key = resolve_api_key(api_key)
        self._chain: Optional[List[str]] = None
        self.calls = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def model_chain(self) -> List[str]:
        if self._chain is not None:
            return self._chain
        chain: List[str] = []
        if self.config.model:
            chain.append(self.config.model)
        chain.append(AUTO_ROUTER)
        if self.config.enable_discovery:
            chain += discover_free_models(self.api_key)
        chain += list(STATIC_FALLBACK_MODELS)
        seen, ordered = set(), []
        for m in chain:
            if m and m not in seen:
                seen.add(m)
                ordered.append(m)
        self._chain = ordered
        return ordered

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:                          # noqa: BLE001
            raise RuntimeError(
                "The LLM proposer needs the 'openai' package (OpenRouter is "
                "OpenAI-API-compatible). Install it with `pip install openai`, or run "
                "with use_llm=False.") from exc
        return OpenAI(base_url=OPENROUTER_BASE, api_key=self.api_key,
                      timeout=self.config.timeout)

    def propose(self, n: int, target_type: str = "generic",
                target_class: Optional[str] = None, target_stem: Optional[str] = None,
                avoid_names: Sequence[str] = (),
                rejected_this_run: Sequence[str] = ()) -> LLMResult:
        result = LLMResult()
        if not self.available:
            result.error = "no API key (set OPENROUTER_API_KEY); skipping the LLM proposer"
            return result

        prompt = build_prompt(target_type, target_class, target_stem, n,
                              avoid_names, rejected_this_run)
        suffix = (target_stem or "").lstrip("-") if target_type == "generic" else ""
        t0 = time.perf_counter()

        try:
            client = self._client()
        except RuntimeError as exc:
            result.error = str(exc)
            return result

        for model in self.model_chain()[: self.config.max_attempts]:
            result.attempts.append(model)
            try:
                self.calls += 1
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                text = (resp.choices[0].message.content or "") if resp.choices else ""
                names = _parse_names(text, n, suffix)
                if names:
                    result.names = names
                    result.model_used = model
                    result.latency_s = round(time.perf_counter() - t0, 2)
                    return result
                result.error = f"{model}: returned no parseable candidates"
            except Exception as exc:                        # noqa: BLE001
                result.error = f"{model}: {type(exc).__name__}: {exc}"[:200]
                time.sleep(self.config.cooldown_s)          # usually a rate limit

        result.latency_s = round(time.perf_counter() - t0, 2)
        return result
