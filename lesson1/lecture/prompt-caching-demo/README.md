# Anthropic Prompt Caching Experiment

A controlled A/B experiment that runs the **same eight-turn compliance review conversation** twice against the Anthropic Messages API — once without caching and once with automatic prompt caching — so you can illustrate cost, token accounting, and latency differences.

The fixtures are intentionally heavyweight: a ~10k-token static prefix (system prompt, twelve tool definitions, and an expanded policy document) plus multi-part audit questions that produce long structured answers.

## Hypothesis

When a large static prefix (system prompt + tools + reference document) is reused across multiple turns, enabling `cache_control` will:

1. **Turn 1:** incur a cache **write** (`cache_creation_input_tokens` > 0)
2. **Turns 2–8:** shift most prefix tokens to **reads** (`cache_read_input_tokens` > 0)
3. **Reduce** estimated input cost and (often) latency on turns 2–8

Without `cache_control`, all three cache fields stay at zero every turn — and the full growing context is billed at full input price each time.

## Experimental design

### Independent variable

| Arm | Configuration |
|-----|---------------|
| **A (baseline)** | No `cache_control` anywhere |
| **B (treatment)** | Top-level `cache_control={"type": "ephemeral"}` (automatic caching) |

### Held constant

- Model (`claude-sonnet-4-6` by default)
- System prompt, reference document, tool definitions (byte-identical)
- Eight user messages (same order, same text)
- `max_tokens` (1024 by default), temperature defaults
- Fresh conversation per arm (no cross-contamination)

### Dependent variables (per turn)

| Metric | Source |
|--------|--------|
| `cache_creation_input_tokens` | Cache write size |
| `cache_read_input_tokens` | Cache hit size |
| `input_tokens` | Uncached suffix (after breakpoint) |
| `output_tokens` | Generated tokens |
| Latency (seconds) | Wall clock per request |
| Estimated input cost | Derived from usage fields |

### Prompt structure

```
tools[]           ─┐
system[0] instructions │
system[1] document     ├── static prefix (~10k+ tokens)
messages (growing)  ───┘── breakpoint moves forward each turn (automatic mode)
```

The fixtures are sized well above Sonnet's ~1,024-token minimum cache threshold.

## Expected results

| Turn | No cache | With cache |
|------|----------|------------|
| 1 | create=0, read=0 | create ≈ prefix size, read=0 |
| 2 | create=0, read=0 | create=0, read ≈ prior prefix |
| 3–8 | same as turn 2 — full context re-billed | read grows as conversation history grows |
| Cost | full input price every turn | ~0.1× on cached prefix after turn 1 |

**Important:** Turn 1 with caching may cost *slightly more* than baseline (1.25× write multiplier on cached tokens). Savings appear from turn 2 onward and compound as the conversation grows.

## Run it

```bash
cd lesson1/lecture/prompt-caching-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python run_experiment.py --output results.json
```

The from-scratch caching engine lab is in
[`../../lab/prompt-caching-lab/`](../../lab/prompt-caching-lab/).

Options:

- `--baseline-only` — run only the no-cache arm
- `--caching-only` — run only the cached arm
- `--model claude-sonnet-4-6` — override model
- `--max-tokens 1024` — cap response length (default 1024)
- `--cooldown 2` — pause between turns

**Note:** A full run (both arms, 8 turns) makes 16 API calls and will incur real API cost. The no-cache arm in particular re-processes the entire prefix on every turn.

## Interpreting output

Column key:

- **Create** — tokens written to cache this request
- **Read** — tokens served from cache
- **Fresh** — tokens after the cache breakpoint (new content only)
- **Hit%** — `read / (create + read + fresh)`

For a successful demo, audience should see:

1. Baseline: Create and Read always 0; per-turn cost climbs as conversation history grows
2. Cached: Create spikes on turn 1, Read dominates turns 2–8
3. Summary: substantial input cost reduction across all eight turns (often 50%+)

## Presentation tips

1. **Run cached arm first** if demoing live — turn 2 latency drop is the visual "aha."
2. **Show the JSON usage block** from turn 2 side-by-side.
3. **Explain `input_tokens` nuance** — it is *not* total input on Anthropic.
4. **Point at turn 5–8 on the baseline table** — costs accelerate because the static prefix is re-billed every time.
5. **Deliberately break cache** (optional encore): add a timestamp to system prompt and re-run turn 2; reads drop to zero.

## Variant: explicit breakpoints

For finer-grained illustration, replace automatic caching with explicit markers:

```python
system=[
    {"type": "text", "text": instructions},
    {"type": "text", "text": document, "cache_control": {"type": "ephemeral"}},
]
tools=[*tools[:-1], {**tools[-1], "cache_control": {"type": "ephemeral"}}]
```

This makes it obvious *which* layers are cached independently.

## Files

- `fixtures/system_prompt.txt` — analyst instructions (expanded)
- `fixtures/reference_document.txt` — ~8k token policy doc with control matrix and appendices
- `fixtures/tools.json` — twelve tool definitions
- `fixtures/user_turns.json` — eight multi-part audit questions
- `run_experiment.py` — experiment runner

## References

- [Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Anthropic pricing](https://www.anthropic.com/pricing)
