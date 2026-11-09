import time
import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

QUESTION = "What is the capital of France?"
MESH_API_KEY = os.getenv("MESH_API_KEY")

# Note: this is the STANDARD OpenAI Protocol SDK
client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key=MESH_API_KEY,
)
PROMPTS = [
    "Summarize in 2 sentences why prefix caching lowers LLM inference cost.",
    "Explain the difference between prefill and decode in one short paragraph.",
    "List 3 reasons an agent's context grows across steps. Be brief.",
    "What is speculative decoding? Answer in under 50 words.",
    "Why is output pricing usually higher than input pricing? 2 sentences.",
]

for alias in ["mesh-gpt-mini", "mesh-claude-haiku"]:
    for i, prompt in enumerate(PROMPTS):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=alias,
            messages=[{"role": "user", "content": prompt}],
            # Metadata flows through the mesh into Langfuse:
            extra_body={
                "metadata": {
                    "generation_name": f"lab4-q{i}",
                    "session_id": "lab1-cost-experiment",
                    "tags": ["lab1", alias],
                }
            },
        )
        print(f"{alias} q{i}: {time.time()-t0:.2f}s, "
              f"{resp.usage.prompt_tokens}+{resp.usage.completion_tokens} tokens")
