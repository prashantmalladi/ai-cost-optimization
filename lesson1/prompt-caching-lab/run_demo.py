#!/usr/bin/env python3
"""
Replay the parent project's 8-turn compliance-review conversation through YOUR
from-scratch PrefixCache (PROVIDED — do not edit).

No API key and no network needed: the prompts come from ./fixtures and an
expensive-but-fake MockModel provides latency. The point is to watch your own
engine reproduce the create / read / fresh shape you saw against the real
Anthropic API, and quantify the cost + latency savings.

    python run_demo.py                 # both arms, default settings
    python run_demo.py --block-size 32 # bigger cache blocks
    python run_demo.py --no-sleep      # skip latency simulation (fast)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import importlib
import os

from mock_model import MockModel

# Defaults to your implementation; instructors may point this at the reference
# with PROMPT_CACHE_IMPL=reference_solution python run_demo.py
_impl = importlib.import_module(os.environ.get("PROMPT_CACHE_IMPL", "prompt_cache"))
PrefixCache = _impl.PrefixCache
Usage = _impl.Usage
estimate_input_cost = _impl.estimate_input_cost
tokenize = _impl.tokenize

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_static_prefix() -> str:
    """Same static prefix as the parent experiment: system + tools + doc."""
    system = (FIXTURES / "system_prompt.txt").read_text()
    document = (FIXTURES / "reference_document.txt").read_text()
    tools = (FIXTURES / "tools.json").read_text()
    return (
        f"{system}\n\nTOOLS:\n{tools}\n\n"
        f"REFERENCE DOCUMENT (authoritative for this session):\n\n{document}"
    )


def load_turns() -> list[str]:
    return json.loads((FIXTURES / "user_turns.json").read_text())


def no_cache_usage(token_ids: list[int]) -> Usage:
    """The baseline arm: every token is processed fresh, every turn."""
    total = len(token_ids)
    return Usage(
        cache_read_tokens=0,
        cache_creation_tokens=0,
        fresh_tokens=total,
        total_tokens=total,
        estimated_input_cost_usd=estimate_input_cost(total, 0, 0),
    )


def print_table(title: str, rows: list[tuple[int, float, Usage, int]]) -> None:
    print(f"\n{title}")
    header = (
        f"{'Turn':>4} {'Latency':>9} {'Create':>8} {'Read':>8} "
        f"{'Fresh':>8} {'TotalIn':>8} {'Out':>5} {'Cost$':>10} {'Hit%':>6}"
    )
    print(header)
    print("-" * len(header))
    for turn, latency, u, out in rows:
        print(
            f"{turn:>4} {latency:>8.3f}s {u.cache_creation_tokens:>8} "
            f"{u.cache_read_tokens:>8} {u.fresh_tokens:>8} {u.total_tokens:>8} "
            f"{out:>5} {u.estimated_input_cost_usd:>10.6f} "
            f"{u.cache_hit_rate * 100:>5.1f}%"
        )


def run_arm(
    *,
    cache: PrefixCache | None,
    static_prefix: str,
    turns: list[str],
    model: MockModel,
    output_tokens: int,
    sleep: bool,
) -> list[tuple[int, float, Usage, int]]:
    """Run all turns. `cache=None` means the baseline (no caching) arm."""
    transcript = static_prefix
    rows: list[tuple[int, float, Usage, int]] = []
    clock = 0.0
    for turn, user_text in enumerate(turns, start=1):
        transcript += f"\n\nUSER: {user_text}\n\nASSISTANT:"
        token_ids = tokenize(transcript)

        if cache is None:
            usage = no_cache_usage(token_ids)
        else:
            usage = cache.process(token_ids, now=clock)

        resp = model.generate(usage, output_tokens, sleep=sleep)
        rows.append((turn, resp.latency_s, usage, output_tokens))

        # Simulate a fixed think-time between turns; well under the TTL so the
        # cached prefix survives across the conversation.
        clock += resp.latency_s + 1.0
        transcript += f" (assistant produced ~{output_tokens} tokens)"
    return rows


def summarize(rows: list[tuple[int, float, Usage, int]]) -> dict:
    return {
        "total_cost_usd": round(sum(u.estimated_input_cost_usd for _, _, u, _ in rows), 6),
        "total_reads": sum(u.cache_read_tokens for _, _, u, _ in rows),
        "total_writes": sum(u.cache_creation_tokens for _, _, u, _ in rows),
        "mean_latency_s": round(statistics.mean(l for _, l, _, _ in rows), 3),
        "turn2plus_mean_latency_s": round(
            statistics.mean(l for t, l, _, _ in rows if t >= 2), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--min-cacheable", type=int, default=1024)
    parser.add_argument("--ttl", type=float, default=300.0)
    parser.add_argument("--max-blocks", type=int, default=100_000)
    parser.add_argument("--output-tokens", type=int, default=400)
    parser.add_argument("--no-sleep", action="store_true", help="Skip latency simulation")
    args = parser.parse_args()

    static_prefix = build_static_prefix()
    turns = load_turns()
    model = MockModel()
    sleep = not args.no_sleep

    prefix_tokens = len(tokenize(static_prefix))
    print("From-scratch prompt cache demo")
    print(f"  Static prefix:    ~{prefix_tokens:,} tokens")
    print(f"  Turns:            {len(turns)}")
    print(f"  Block size:       {args.block_size} tokens")
    print(f"  Min cacheable:    {args.min_cacheable} tokens")
    print(f"  TTL:              {args.ttl:.0f}s")

    baseline = run_arm(
        cache=None,
        static_prefix=static_prefix,
        turns=turns,
        model=model,
        output_tokens=args.output_tokens,
        sleep=sleep,
    )
    print_table("=== Arm A: WITHOUT prompt caching (baseline) ===", baseline)

    cache = PrefixCache(
        block_size=args.block_size,
        min_cacheable_tokens=args.min_cacheable,
        ttl_seconds=args.ttl,
        max_blocks=args.max_blocks,
    )
    cached = run_arm(
        cache=cache,
        static_prefix=static_prefix,
        turns=turns,
        model=model,
        output_tokens=args.output_tokens,
        sleep=sleep,
    )
    print_table("=== Arm B: WITH your PrefixCache ===", cached)

    a, b = summarize(baseline), summarize(cached)
    cost_savings = 1 - b["total_cost_usd"] / a["total_cost_usd"]
    lat_savings = 1 - b["turn2plus_mean_latency_s"] / a["turn2plus_mean_latency_s"]

    print("\n=== Summary ===")
    print(json.dumps({"baseline": a, "cached": b}, indent=2))
    print(f"\nInput cost reduction:           {cost_savings * 100:.1f}%")
    print(f"Latency improvement (turns 2+): {lat_savings * 100:.1f}%")
    print(f"Blocks resident in cache:       {len(cache)}")


if __name__ == "__main__":
    main()
