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

# List your model aliases here
MODEL_ALIASES = [
    "mesh-gpt-mini",
    "mesh-claude-haiku"
]

for model in MODEL_ALIASES:
    print(f"\n--- Model: {model} ---")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": QUESTION}
            ],
        )

        answer = response.choices[0].message.content
        print(answer)

    except Exception as e:
        print(f"Error: {e}")