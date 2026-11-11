# Prompt Caching Lab

Build a `PrefixCache` engine from scratch — prefix-dependent block hashing,
longest-prefix matching, TTL expiry, LRU eviction, and cost accounting — until
it reproduces the create / read / fresh numbers from the lecture demo.

The live Anthropic A/B experiment lives in
[`../../lecture/prompt-caching-demo/`](../../lecture/prompt-caching-demo/).
Work through that first if you have not seen the 8-turn compliance-review
conversation against the real API. **This lab needs no API key or network.**

## The big idea

A long conversation re-sends the same static prefix on every turn. Without
caching, the provider re-processes that whole prefix each time — you pay full
input price for it again and again. With caching, the prefix is processed once
(a cache **write**), then served cheaply from cache on later turns (a **read**).

```
tokens:  [ system + tools + document ][ turn 1 ][ turn 2 ] ...
          \___________________________/
           static prefix — cache it once, read it cheaply thereafter
```

- A cache **read** costs ~0.1× the base input price.
- A cache **write** costs ~1.25× (so turn 1 is slightly *more* expensive).
- Savings appear from turn 2 onward and compound as the conversation grows.

## Setup

```bash
cd lesson1/lab/prompt-caching-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # just pytest

# Fill in the TODOs in prompt_cache.py, then:
pytest -v                          # your definition of "done"
python run_demo.py                 # replays the 8-turn run through your engine
```

A correct implementation prints a write spike on turn 1, ~99% read hit rate on
turns 2–8, and a 70%+ cost reduction — the same shape as the live API demo.
The method contracts are in `prompt_cache.py`.

## Layout

```
prompt-caching-lab/
├── README.md              # you are here
├── prompt_cache.py        # <- start here (stubs to implement)
├── test_prompt_cache.py   # test suite
├── reference_solution.py  # complete solution (don't peek early)
├── mock_model.py          # fake "expensive" model for latency wins
├── run_demo.py            # replays the lecture conversation through your engine
├── fixtures/              # same prompts as the live demo (offline replay)
└── requirements.txt
```

## Prerequisites

- Python 3.10+

## References

- [Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Anthropic pricing](https://www.anthropic.com/pricing)
- Lecture demo: [`../../lecture/prompt-caching-demo/`](../../lecture/prompt-caching-demo/)
