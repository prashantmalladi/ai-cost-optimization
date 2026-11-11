#!/usr/bin/env python3
"""
Anthropic prompt caching A/B experiment.

Runs the same multi-turn compliance review conversation twice:
  - WITHOUT cache_control (baseline)
  - WITH cache_control (automatic top-level caching)

Records per-turn usage, latency, and estimated input cost.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import anthropic
except ImportError:
    print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

FIXTURES = Path(__file__).parent / "fixtures"


def log(msg: str = "") -> None:
    """Print progress immediately (unbuffered for long-running API calls)."""
    print(msg, flush=True)


# Claude Sonnet 4.6 list pricing (USD per million tokens)
INPUT_PRICE_PER_M = 3.00
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


@dataclass
class TurnMetrics:
    condition: str
    turn: int
    latency_s: float
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_input_tokens: int
    estimated_input_cost_usd: float

    @property
    def cache_hit_rate(self) -> float:
        if self.total_input_tokens == 0:
            return 0.0
        return self.cache_read_input_tokens / self.total_input_tokens


def load_fixtures() -> tuple[str, str, list[dict[str, Any]], list[str]]:
    tools = json.loads((FIXTURES / "tools.json").read_text())
    system_prompt = (FIXTURES / "system_prompt.txt").read_text()
    document = (FIXTURES / "reference_document.txt").read_text()
    user_turns = json.loads((FIXTURES / "user_turns.json").read_text())
    return tools, system_prompt, document, user_turns


def build_system_blocks(system_prompt: str, document: str) -> list[dict[str, Any]]:
    """Static prefix: instructions + full reference document."""
    return [
        {"type": "text", "text": system_prompt},
        {
            "type": "text",
            "text": (
                "REFERENCE DOCUMENT (authoritative for this session):\n\n"
                f"{document}"
            ),
        },
    ]


def estimate_input_cost(
    input_tokens: int,
    cache_creation: int,
    cache_read: int,
) -> float:
    per_token = INPUT_PRICE_PER_M / 1_000_000
    return (
        input_tokens * per_token
        + cache_creation * per_token * CACHE_WRITE_MULTIPLIER
        + cache_read * per_token * CACHE_READ_MULTIPLIER
    )


def run_conversation(
    client: anthropic.Anthropic,
    *,
    model: str,
    condition: str,
    caching_enabled: bool,
    system_blocks: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    user_turns: list[str],
    max_tokens: int,
    cooldown_s: float,
) -> list[TurnMetrics]:
    messages: list[dict[str, Any]] = []
    results: list[TurnMetrics] = []
    total_turns = len(user_turns)

    for turn_idx, user_text in enumerate(user_turns, start=1):
        preview = user_text.replace("\n", " ").strip()
        if len(preview) > 72:
            preview = preview[:69] + "..."
        log(f"  Turn {turn_idx}/{total_turns}: calling API — \"{preview}\"")

        messages.append({"role": "user", "content": user_text})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": tools,
            "system": system_blocks,
            "messages": messages,
        }
        
        if caching_enabled:
            kwargs["cache_control"] = {"type": "ephemeral"}

        start = time.perf_counter()
        
        response = client.messages.create(**kwargs)

        latency = time.perf_counter() - start

        usage = response.usage
        cache_create = usage.cache_creation_input_tokens or 0
        cache_read = usage.cache_read_input_tokens or 0
        input_tokens = usage.input_tokens or 0
        total_input = cache_create + cache_read + input_tokens

        metrics = TurnMetrics(
            condition=condition,
            turn=turn_idx,
            latency_s=round(latency, 3),
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens or 0,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            total_input_tokens=total_input,
            estimated_input_cost_usd=round(
                estimate_input_cost(input_tokens, cache_create, cache_read), 6
            ),
        )
        results.append(metrics)

        log(
            f"  Turn {turn_idx}/{total_turns}: "
            f"done in {latency:.1f}s "
            f"(create={cache_create:,}, read={cache_read:,}, "
            f"fresh={input_tokens:,}, out={metrics.output_tokens:,})"
        )

        # Append assistant reply for multi-turn continuity
        assistant_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        messages.append({"role": "assistant", "content": assistant_text})

        if cooldown_s > 0 and turn_idx < total_turns:
            log(f"  Waiting {cooldown_s:.1f}s before next turn...")
            time.sleep(cooldown_s)

    return results


def print_table(rows: list[TurnMetrics]) -> None:
    header = (
        f"{'Cond':<12} {'Turn':>4} {'Latency':>8} "
        f"{'Create':>8} {'Read':>8} {'Fresh':>8} "
        f"{'TotalIn':>8} {'Out':>6} {'Cost$':>10} {'Hit%':>6}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in rows:
        print(
            f"{r.condition:<12} {r.turn:>4} {r.latency_s:>7.3f}s "
            f"{r.cache_creation_input_tokens:>8} {r.cache_read_input_tokens:>8} "
            f"{r.input_tokens:>8} {r.total_input_tokens:>8} "
            f"{r.output_tokens:>6} {r.estimated_input_cost_usd:>10.6f} "
            f"{r.cache_hit_rate * 100:>5.1f}%",
            flush=True,
        )


def summarize(rows: list[TurnMetrics], label: str) -> dict[str, Any]:
    subset = [r for r in rows if r.condition == label]
    return {
        "condition": label,
        "turns": len(subset),
        "total_input_tokens": sum(r.total_input_tokens for r in subset),
        "total_cache_reads": sum(r.cache_read_input_tokens for r in subset),
        "total_cache_writes": sum(r.cache_creation_input_tokens for r in subset),
        "total_fresh_input": sum(r.input_tokens for r in subset),
        "total_output_tokens": sum(r.output_tokens for r in subset),
        "total_estimated_input_cost_usd": round(
            sum(r.estimated_input_cost_usd for r in subset), 6
        ),
        "mean_latency_s": round(statistics.mean(r.latency_s for r in subset), 3),
        "turn2_plus_mean_latency_s": round(
            statistics.mean(r.latency_s for r in subset if r.turn >= 2), 3
        )
        if any(r.turn >= 2 for r in subset)
        else None,
        "turn2_plus_mean_cache_hit_rate": round(
            statistics.mean(r.cache_hit_rate for r in subset if r.turn >= 2), 4
        )
        if any(r.turn >= 2 for r in subset)
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        help="Model ID (default: claude-sonnet-4-6)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.0,
        help="Seconds between turns (avoid rate limits)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON results to this path",
    )
    parser.add_argument(
        "--caching-only",
        action="store_true",
        help="Skip baseline (no-cache) run",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip caching run",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY before running.", file=sys.stderr)
        sys.exit(1)

    tools, system_prompt, document, user_turns = load_fixtures()
    system_blocks = build_system_blocks(system_prompt, document)
    client = anthropic.Anthropic()

    arms: list[str] = []
    if not args.caching_only:
        arms.append("baseline (no cache)")
    if not args.baseline_only:
        arms.append("caching enabled")

    log()
    log("Anthropic prompt caching experiment")
    log(f"  Model:      {args.model}")
    log(f"  User turns: {len(user_turns)}")
    log(f"  Tools:      {len(tools)}")
    log(f"  Max tokens: {args.max_tokens} per response")
    static_prefix_est = (len(system_prompt) + len(document) + len(json.dumps(tools))) // 4
    log(f"  Static prefix (est.): ~{static_prefix_est:,} tokens (system + doc + tools)")
    log(f"  Arms:       {', '.join(arms)}")
    log(f"  Cooldown:   {args.cooldown:.1f}s between turns")
    log()

    all_rows: list[TurnMetrics] = []
    arm_num = 0

    if not args.caching_only:
        arm_num += 1
        log(f"=== Arm {arm_num}/{len(arms)}: WITHOUT prompt caching ===")
        baseline = run_conversation(
            client,
            model=args.model,
            condition="no_cache",
            caching_enabled=False,
            system_blocks=system_blocks,
            tools=tools,
            user_turns=user_turns,
            max_tokens=args.max_tokens,
            cooldown_s=args.cooldown,
        )
        all_rows.extend(baseline)
        log()
        log("Results (no cache):")
        print_table(baseline)
        log()

    if not args.baseline_only:
        arm_num += 1
        log(f"=== Arm {arm_num}/{len(arms)}: WITH prompt caching (automatic) ===")
        cached = run_conversation(
            client,
            model=args.model,
            condition="with_cache",
            caching_enabled=True,
            system_blocks=system_blocks,
            tools=tools,
            user_turns=user_turns,
            max_tokens=args.max_tokens,
            cooldown_s=args.cooldown,
        )
        all_rows.extend(cached)
        log()
        log("Results (with cache):")
        print_table(cached)
        log()

    summaries = []
    if not args.caching_only:
        summaries.append(summarize(all_rows, "no_cache"))
    if not args.baseline_only:
        summaries.append(summarize(all_rows, "with_cache"))

    log("=== Summary ===")
    for s in summaries:
        print(json.dumps(s, indent=2), flush=True)

    if len(summaries) == 2:
        no_cache, with_cache = summaries
        savings = (
            1
            - with_cache["total_estimated_input_cost_usd"]
            / no_cache["total_estimated_input_cost_usd"]
        )
        log()
        log(
            f"Estimated input cost reduction (cached vs baseline): {savings * 100:.1f}%"
        )
        if (
            with_cache["turn2_plus_mean_latency_s"]
            and no_cache["turn2_plus_mean_latency_s"]
        ):
            lat_delta = (
                1
                - with_cache["turn2_plus_mean_latency_s"]
                / no_cache["turn2_plus_mean_latency_s"]
            )
            log(
                f"Mean latency improvement turns 2-{len(user_turns)}: "
                f"{lat_delta * 100:.1f}%"
            )

    if args.output:
        payload = {
            "model": args.model,
            "turns": [asdict(r) for r in all_rows],
            "summaries": summaries,
        }
        args.output.write_text(json.dumps(payload, indent=2))
        log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
