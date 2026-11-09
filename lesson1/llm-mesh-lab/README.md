# LLM Mesh Lab

A small lab that puts one local "front door" (a proxy) in front of several AI providers. Your code talks to the front door as if it were OpenAI. The front door forwards each request to OpenAI or Anthropic and records what happened.

## How it fits together

```
┌─────────────────┐
│  client_test.py │──┐
└─────────────────┘  │     ┌──────────────────────────────┐     ┌───────────────────────────┐
                     ├────►│  LiteLLM proxy               │────►│  OpenAI (gpt-4o-mini)     │
┌─────────────────┐  │     │  localhost:4000              │     └───────────────────────────┘
│  workload.py    │──┘     │  config.yaml: aliases + key  │
└─────────────────┘        │                              │     ┌───────────────────────────┐
                           │                              │────►│  Anthropic (Claude Haiku) │
                           └──────────────┬───────────────┘     └───────────────────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │  OpenTelemetry (OTLP)        │
                           │  sends traces of each call   │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │  Langfuse                    │
                           │  logs: cost, speed, tokens   │
                           └──────────────────────────────┘
```

## What's in here

- `config.yaml`: sets up the LiteLLM proxy. It defines two model nicknames (aliases): `mesh-gpt-mini` (OpenAI gpt-4o-mini) and `mesh-claude-haiku` (Anthropic Claude Haiku 4.5). It also sets the proxy password (`MESH_API_KEY`) and turns on Langfuse tracing (the `langfuse_otel` callback, which sends traces over OpenTelemetry, a standard way to export "what happened" data).
- `client_test.py`: asks both models "What is the capital of France?" and prints the answers. Use it to check the setup works.
- `workload.py`: sends 5 short questions to each model and prints the time taken and token counts. It tags each call with metadata so you can compare cost and speed in Langfuse.
- `pyproject.toml`, `uv.lock`: Python dependencies, managed with `uv`. This includes the OpenTelemetry packages (`opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`) that the proxy uses to ship traces to Langfuse.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- Environment variables, set in a `.env` file (never commit it):
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `MESH_API_KEY`
  - `LANGFUSE_PUBLIC_KEY`
  - `LANGFUSE_SECRET_KEY`
  - `LANGFUSE_HOST`

## How to run

1. Install dependencies: `uv sync`
2. Start the proxy: `uv run litellm --config config.yaml --port 4000`
3. In a second terminal, check it works: `uv run client_test.py`
4. Then run the workload: `uv run workload.py`

## Why it matters

Your app only knows one address and one API style. You can swap models or providers by editing `config.yaml`, with no code changes. Every call is logged, so you can see where your money goes.
