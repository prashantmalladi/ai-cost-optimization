#!/usr/bin/env python3
"""
EXERCISE: Implement a prompt-caching engine from scratch.

In the parent project you ran a *black-box* experiment: you set
`cache_control={"type": "ephemeral"}` and watched Anthropic's API report
`cache_creation_input_tokens` and `cache_read_input_tokens`. You never saw
*how* the provider decided what to cache, what counted as a hit, or how the
TTL and eviction worked.

In this exercise you build that machinery yourself. By the end you will have a
`PrefixCache` that, given a stream of growing prompts, reproduces the exact
create / read / fresh accounting you observed against the real API — and you
will understand precisely why prompt caching behaves the way it does.

------------------------------------------------------------------------------
WHAT YOU IMPLEMENT
------------------------------------------------------------------------------
You only need to edit the five methods marked with `TODO(student)` below:

  1. PrefixCache._block_hashes   -> prefix-dependent block hashing
  2. PrefixCache.lookup          -> longest matching cached prefix (+ TTL)
  3. PrefixCache.store           -> write new blocks, refresh hits
  4. PrefixCache._evict          -> LRU eviction under a block budget
  5. PrefixCache.process         -> per-request accounting (read/write/fresh)

Everything else (tokenizer, dataclasses, pricing constants) is provided.

Run `pytest` to check your work. Run `python run_demo.py` to watch your
engine reproduce the 8-turn compliance-review table from the parent project.

Mental model of provider-side prompt caching:

    tokens:  [ b0 ][ b1 ][ b2 ][ b3 ]  ...tail (partial block)
              \____________________/   \________/
               cacheable, block-aligned  always fresh

  - Caching happens in fixed-size BLOCKS. Only complete blocks are cacheable;
    the trailing partial block is always re-processed ("fresh").
  - A block is identified by a hash that CHAINS in every preceding block, so a
    block only "matches" if the entire prefix before it is byte-identical.
  - On a request we find the longest run of leading blocks already in the cache
    (a "read"/hit), write the new blocks after it (a "write"/create), and the
    remaining tail tokens are "fresh".
  - Entries are ephemeral: they expire after `ttl_seconds`, and reading an
    entry refreshes its TTL (sliding window, like Anthropic's 5-minute TTL).
  - Memory is bounded: when the number of cached blocks exceeds `max_blocks`,
    the least-recently-used blocks are evicted.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------- #
# Pricing constants (Claude Sonnet 4.6 list pricing, USD per million tokens).
# Identical to the parent project so your numbers line up.
# --------------------------------------------------------------------------- #
INPUT_PRICE_PER_M = 3.00
CACHE_WRITE_MULTIPLIER = 1.25  # writing to cache costs 1.25x base input price
CACHE_READ_MULTIPLIER = 0.10   # reading from cache costs 0.10x base input price


# --------------------------------------------------------------------------- #
# Tokenizer (PROVIDED — do not edit).
#
# A real tokenizer maps text to integer token ids. We approximate one with a
# deterministic word/punctuation splitter so the exercise has no heavy
# dependencies. `zlib.crc32` gives a stable id across processes/runs, which
# keeps the demo reproducible.
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SEED_HASH = hashlib.sha256(b"prompt-cache-seed").hexdigest()


def tokenize(text: str) -> list[int]:
    """Deterministically convert text into a list of integer token ids."""
    return [zlib.crc32(tok.encode("utf-8")) for tok in _TOKEN_RE.findall(text)]


# --------------------------------------------------------------------------- #
# Data structures (PROVIDED — do not edit).
# --------------------------------------------------------------------------- #
@dataclass
class BlockRef:
    """A single complete, block-aligned chunk of the prompt prefix.

    `chain_hash` depends on this block's tokens AND every preceding block, so
    two prompts share a `chain_hash` at position i only if their first
    (i+1) blocks are identical.
    """

    index: int           # 0-based position in the prefix
    chain_hash: str      # prefix-dependent identifier
    n_tokens: int        # tokens contained in this block
    cumulative_tokens: int  # tokens from the start through the end of this block


@dataclass
class CacheEntry:
    """A cached block living in the store."""

    chain_hash: str
    n_tokens: int
    created_at: float    # set/refreshed whenever the entry is read or written
    last_used: float     # used for LRU eviction


@dataclass
class Usage:
    """Per-request accounting, mirroring Anthropic's `usage` object."""

    cache_read_tokens: int          # served from cache (Anthropic: cache_read_input_tokens)
    cache_creation_tokens: int      # written to cache (Anthropic: cache_creation_input_tokens)
    fresh_tokens: int               # uncached tail (Anthropic: input_tokens)
    total_tokens: int
    estimated_input_cost_usd: float

    @property
    def cache_hit_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.cache_read_tokens / self.total_tokens


def estimate_input_cost(fresh: int, write: int, read: int) -> float:
    """Cost of the input side of one request, in USD."""
    per_token = INPUT_PRICE_PER_M / 1_000_000
    return (
        fresh * per_token
        + write * per_token * CACHE_WRITE_MULTIPLIER
        + read * per_token * CACHE_READ_MULTIPLIER
    )


# --------------------------------------------------------------------------- #
# The engine you implement.
# --------------------------------------------------------------------------- #
class PrefixCache:
    """A block-based, TTL'd, LRU-evicting prompt prefix cache.

    Parameters
    ----------
    block_size:
        Tokens per cacheable block. Real providers use large blocks; the demo
        uses a small value so behavior is observable on tiny fixtures.
    min_cacheable_tokens:
        Prompts shorter than this are never cached (Anthropic's Sonnet minimum
        is ~1024). Below the threshold: everything is `fresh`.
    ttl_seconds:
        Entry lifetime. Reading an entry refreshes it (sliding TTL).
    max_blocks:
        Capacity. When exceeded, least-recently-used blocks are evicted.
    """

    def __init__(
        self,
        *,
        block_size: int = 16,
        min_cacheable_tokens: int = 0,
        ttl_seconds: float = 300.0,
        max_blocks: int = 10_000,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.min_cacheable_tokens = min_cacheable_tokens
        self.ttl_seconds = ttl_seconds
        self.max_blocks = max_blocks
        self._store: dict[str, CacheEntry] = {}

    # ------------------------------------------------------------------ #
    # Introspection helpers (PROVIDED).
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._store)

    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        return (now - entry.created_at) > self.ttl_seconds

    # ================================================================== #
    # TODO(student) #1: prefix-dependent block hashing
    # ================================================================== #
    def _block_hashes(self, token_ids: list[int]) -> list[BlockRef]:
        """Split `token_ids` into complete, block-aligned chunks and assign each
        a *chained* hash.

        Requirements:
          - Only emit COMPLETE blocks of exactly `self.block_size` tokens. Drop
            the trailing partial block (it is never cacheable).
          - The hash of block i must depend on the tokens of block i AND the
            hash of block i-1, so that a block only matches when its whole
            prefix matches. (Hint: feed the previous chain_hash plus this
            block's token bytes into hashlib.sha256.)
          - Populate `index`, `chain_hash`, `n_tokens` (== block_size), and
            `cumulative_tokens` for each BlockRef.

        Returns an empty list if there are fewer than `block_size` tokens.
        """
        blocks: list[BlockRef] = []
        prev_hash = _SEED_HASH
        n_full = len(token_ids) // self.block_size
        for i in range(n_full):
            chunk = token_ids[i * self.block_size : (i + 1) * self.block_size]
            h = hashlib.sha256()
            h.update(prev_hash.encode("utf-8"))
            h.update(b"|")
            h.update(",".join(map(str, chunk)).encode("utf-8"))
            chain = h.hexdigest()
            blocks.append(
                BlockRef(
                    index=i,
                    chain_hash=chain,
                    n_tokens=self.block_size,
                    cumulative_tokens=(i + 1) * self.block_size,
                )
            )
            prev_hash = chain
        return blocks

    # ================================================================== #
    # TODO(student) #2: longest matching cached prefix
    # ================================================================== #
    def lookup(self, blocks: list[BlockRef], now: float) -> int:
        """Return how many LEADING blocks are already cached and not expired.

        Walk `blocks` from the start. Stop at the first block that is missing
        from the store OR whose entry has expired (see `_is_expired`). Return
        the count of contiguous leading blocks that hit.

        Do NOT mutate the store here — this method only reads.
        """
        matched = 0
        for b in blocks:
            entry = self._store.get(b.chain_hash)
            if entry is None or self._is_expired(entry, now):
                break
            matched += 1
        return matched


    # ================================================================== #
    # TODO(student) #3: write new blocks, refresh hits
    # ================================================================== #
    def store(self, blocks: list[BlockRef], matched_blocks: int, now: float) -> None:
        """Persist the result of a request.

          - For the first `matched_blocks` blocks (cache hits): refresh their
            TTL by updating `created_at` and `last_used` to `now` (sliding TTL).
          - For every block after that (cache writes): insert a new CacheEntry
            with `created_at == last_used == now`.
          - After writing, call `self._evict(now)` to respect `max_blocks`.
        """
        for i, b in enumerate(blocks):
            if i < matched_blocks:
                entry = self._store.get(b.chain_hash)
                if entry is not None:
                    entry.created_at = now
                    entry.last_used = now
            else:
                self._store[b.chain_hash] = CacheEntry(
                    chain_hash=b.chain_hash,
                    n_tokens=b.n_tokens,
                    created_at=now,
                    last_used=now,
                )
        self._evict(now)

    # ================================================================== #
    # TODO(student) #4: LRU eviction
    # ================================================================== #
    def _evict(self, now: float) -> None:
        """Evict least-recently-used entries until `len(self._store) <=
        self.max_blocks`.

        "Least recently used" = smallest `last_used` timestamp.
        """
        while len(self._store) > self.max_blocks:
            lru_key = min(self._store, key=lambda k: self._store[k].last_used)
            del self._store[lru_key]

    # ================================================================== #
    # TODO(student) #5: per-request accounting
    # ================================================================== #
    def process(self, token_ids: list[int], now: float) -> Usage:
        """Run one request through the cache and return its Usage.

        Algorithm:
          1. total = len(token_ids).
          2. If total < self.min_cacheable_tokens: nothing is cached. Return a
             Usage with read=0, write=0, fresh=total (and the matching cost).
             Do not touch the store.
          3. Otherwise compute blocks via `_block_hashes`, find the matched
             prefix via `lookup`, then:
               - read  = total tokens in the matched leading blocks
               - write = total tokens in the blocks after the match
               - fresh = total - read - write   (the trailing partial block)
          4. Persist via `store(...)`.
          5. Cost via `estimate_input_cost(fresh, write, read)`.

        Return a fully-populated Usage.
        """
        total = len(token_ids)
        if total < self.min_cacheable_tokens:
            return Usage(
                cache_read_tokens=0,
                cache_creation_tokens=0,
                fresh_tokens=total,
                total_tokens=total,
                estimated_input_cost_usd=estimate_input_cost(total, 0, 0),
            )
        blocks = self._block_hashes(token_ids)
        matched = self.lookup(blocks, now)
        read = sum(b.n_tokens for b in blocks[:matched])
        write = sum(b.n_tokens for b in blocks[matched:])
        self.store(blocks, matched, now)
        fresh = total - read - write
        return Usage(
            cache_read_tokens=read,
            cache_creation_tokens=write,
            fresh_tokens=fresh,
            total_tokens=total,
            estimated_input_cost_usd=estimate_input_cost(fresh, write, read),
        )


# Convenience: a couple of well-known field()/Iterable imports kept so that
# extending the dataclasses (stretch goals) doesn't require touching imports.
__all__ = [
    "PrefixCache",
    "BlockRef",
    "CacheEntry",
    "Usage",
    "tokenize",
    "estimate_input_cost",
    "INPUT_PRICE_PER_M",
    "CACHE_WRITE_MULTIPLIER",
    "CACHE_READ_MULTIPLIER",
]

_ = (field, Iterable, hashlib)  # referenced for student stretch goals
