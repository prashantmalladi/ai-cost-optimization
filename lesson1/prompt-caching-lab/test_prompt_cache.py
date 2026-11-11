#!/usr/bin/env python3
"""
Test suite for the prompt-caching exercise.

Run with:   pytest -v

These tests are your definition of "done". They import your implementation
from `prompt_cache.py` by default. (Instructors can point the tests at a
reference implementation with `PROMPT_CACHE_IMPL=reference_solution pytest`.)
"""

from __future__ import annotations

import importlib
import os

import pytest

impl = importlib.import_module(os.environ.get("PROMPT_CACHE_IMPL", "prompt_cache"))

PrefixCache = impl.PrefixCache
tokenize = impl.tokenize
estimate_input_cost = impl.estimate_input_cost


def seq(n: int) -> list[int]:
    """A deterministic token sequence of length n (ids 0..n-1)."""
    return list(range(n))


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def test_tokenize_is_deterministic():
    assert tokenize("hello world") == tokenize("hello world")


def test_tokenize_distinguishes_text():
    assert tokenize("hello world") != tokenize("hello there")


def test_tokenize_nonempty():
    assert len(tokenize("the quick brown fox jumps")) == 5


# --------------------------------------------------------------------------- #
# #1 block hashing
# --------------------------------------------------------------------------- #
def test_only_complete_blocks_emitted():
    c = PrefixCache(block_size=4)
    blocks = c._block_hashes(seq(10))  # 2 full blocks (8 tokens), 2 left over
    assert len(blocks) == 2
    assert [b.n_tokens for b in blocks] == [4, 4]
    assert [b.cumulative_tokens for b in blocks] == [4, 8]
    assert [b.index for b in blocks] == [0, 1]


def test_fewer_than_block_size_yields_no_blocks():
    c = PrefixCache(block_size=4)
    assert c._block_hashes(seq(3)) == []


def test_chain_hash_shares_prefix():
    c = PrefixCache(block_size=4)
    a = c._block_hashes(seq(12))
    b = c._block_hashes(seq(8))
    # identical leading tokens -> identical chained hashes for shared blocks
    assert a[0].chain_hash == b[0].chain_hash
    assert a[1].chain_hash == b[1].chain_hash


def test_chain_hash_is_prefix_dependent():
    c = PrefixCache(block_size=4)
    base = c._block_hashes(seq(8))
    # change the very first token: every downstream block hash must change
    altered_tokens = [999] + seq(8)[1:]
    altered = c._block_hashes(altered_tokens)
    assert altered[0].chain_hash != base[0].chain_hash
    assert altered[1].chain_hash != base[1].chain_hash


# --------------------------------------------------------------------------- #
# #2 + #3 + #5 cold/warm behavior
# --------------------------------------------------------------------------- #
def test_cold_request_is_all_write():
    c = PrefixCache(block_size=4)
    u = c.process(seq(10), now=0.0)
    assert u.cache_read_tokens == 0
    assert u.cache_creation_tokens == 8  # two full blocks
    assert u.fresh_tokens == 2           # trailing partial block
    assert u.total_tokens == 10


def test_warm_request_is_all_read():
    c = PrefixCache(block_size=4)
    c.process(seq(10), now=0.0)
    u = c.process(seq(10), now=1.0)
    assert u.cache_read_tokens == 8
    assert u.cache_creation_tokens == 0
    assert u.fresh_tokens == 2


def test_growing_prefix_reads_old_writes_new():
    c = PrefixCache(block_size=4)
    c.process(seq(8), now=0.0)          # caches 2 blocks
    u = c.process(seq(16), now=1.0)     # extends to 4 blocks
    assert u.cache_read_tokens == 8     # first two blocks hit
    assert u.cache_creation_tokens == 8  # two new blocks written
    assert u.fresh_tokens == 0


def test_diverging_prefix_busts_downstream():
    c = PrefixCache(block_size=4)
    c.process(seq(16), now=0.0)
    # same first block, different afterwards (16 tokens total -> 4 blocks)
    altered = seq(4) + [777] + seq(11)
    u = c.process(altered, now=1.0)
    assert u.cache_read_tokens == 4      # only the first block survives
    assert u.cache_creation_tokens == 12  # the other three blocks are re-written


# --------------------------------------------------------------------------- #
# min cacheable threshold
# --------------------------------------------------------------------------- #
def test_below_min_cacheable_is_all_fresh_and_not_stored():
    c = PrefixCache(block_size=4, min_cacheable_tokens=100)
    u = c.process(seq(40), now=0.0)
    assert u.fresh_tokens == 40
    assert u.cache_read_tokens == 0
    assert u.cache_creation_tokens == 0
    assert len(c) == 0
    # a second call still gets no hit
    u2 = c.process(seq(40), now=1.0)
    assert u2.cache_read_tokens == 0


# --------------------------------------------------------------------------- #
# #2 TTL
# --------------------------------------------------------------------------- #
def test_entries_expire_after_ttl():
    c = PrefixCache(block_size=4, ttl_seconds=300)
    c.process(seq(8), now=0.0)
    u = c.process(seq(8), now=301.0)   # past TTL
    assert u.cache_read_tokens == 0
    assert u.cache_creation_tokens == 8


def test_read_refreshes_ttl():
    c = PrefixCache(block_size=4, ttl_seconds=300)
    c.process(seq(8), now=0.0)
    c.process(seq(8), now=200.0)        # hit -> refreshes TTL to t=200
    u = c.process(seq(8), now=450.0)    # 250s after refresh, still alive
    assert u.cache_read_tokens == 8


# --------------------------------------------------------------------------- #
# #4 LRU eviction
# --------------------------------------------------------------------------- #
def test_lru_eviction_under_budget():
    c = PrefixCache(block_size=4, max_blocks=2)
    c.process(seq(8), now=0.0)                      # A: 2 blocks resident
    other = [111, 222, 333, 444, 555, 666, 777, 888]
    c.process(other, now=1.0)                       # B: 2 new blocks -> evict A
    assert len(c) == 2
    u = c.process(seq(8), now=2.0)                  # A should be gone
    assert u.cache_read_tokens == 0


def test_recently_used_survives_eviction():
    c = PrefixCache(block_size=4, max_blocks=4)
    c.process(seq(8), now=0.0)                       # A blocks
    other = [111, 222, 333, 444, 555, 666, 777, 888]
    c.process(other, now=1.0)                        # B blocks (cache now full: 4)
    c.process(seq(8), now=2.0)                        # touch A -> A most recent
    third = [9, 8, 7, 6, 5, 4, 3, 2]
    c.process(third, now=3.0)                         # C blocks -> evict LRU (B)
    assert c.process(seq(8), now=4.0).cache_read_tokens == 8  # A survived


# --------------------------------------------------------------------------- #
# #5 cost accounting
# --------------------------------------------------------------------------- #
def test_cost_matches_formula():
    c = PrefixCache(block_size=4)
    c.process(seq(8), now=0.0)
    u = c.process(seq(12), now=1.0)   # read=8, write=4, fresh=0
    expected = estimate_input_cost(u.fresh_tokens, u.cache_creation_tokens, u.cache_read_tokens)
    assert u.estimated_input_cost_usd == pytest.approx(expected)


def test_reads_are_cheaper_than_writes():
    c = PrefixCache(block_size=4)
    cold = c.process(seq(40), now=0.0)   # all write
    warm = c.process(seq(40), now=1.0)   # all read
    assert warm.estimated_input_cost_usd < cold.estimated_input_cost_usd


def test_hit_rate():
    c = PrefixCache(block_size=4)
    c.process(seq(8), now=0.0)
    u = c.process(seq(8), now=1.0)
    assert u.cache_hit_rate == pytest.approx(8 / 8)
