#!/usr/bin/env python3
"""
A tiny stand-in for an expensive LLM (PROVIDED — do not edit).

Real prompt caching saves money AND latency because cached prefix tokens skip
the prefill compute. We model that here: "processing" a token costs wall-clock
time, but cached (read) tokens are ~10x cheaper to process than fresh/written
tokens — mirroring the cost multipliers.

This lets `run_demo.py` show a real latency drop on cached turns without
calling (or paying for) a real API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from prompt_cache import CACHE_READ_MULTIPLIER, Usage


@dataclass
class ModelResponse:
    latency_s: float
    output_tokens: int


class MockModel:
    """Latency is proportional to the *effective* prefill work implied by a
    Usage object: fresh + write tokens at full price, read tokens at 0.1x.
    """

    def __init__(
        self,
        *,
        seconds_per_prefill_token: float = 5e-5,
        seconds_per_output_token: float = 2e-4,
        base_latency_s: float = 0.05,
    ) -> None:
        self.seconds_per_prefill_token = seconds_per_prefill_token
        self.seconds_per_output_token = seconds_per_output_token
        self.base_latency_s = base_latency_s

    def effective_prefill_tokens(self, usage: Usage) -> float:
        return (
            usage.fresh_tokens
            + usage.cache_creation_tokens
            + usage.cache_read_tokens * CACHE_READ_MULTIPLIER
        )

    def generate(self, usage: Usage, output_tokens: int, *, sleep: bool = True) -> ModelResponse:
        prefill = self.effective_prefill_tokens(usage)
        latency = (
            self.base_latency_s
            + prefill * self.seconds_per_prefill_token
            + output_tokens * self.seconds_per_output_token
        )
        if sleep:
            time.sleep(latency)
        return ModelResponse(latency_s=latency, output_tokens=output_tokens)
