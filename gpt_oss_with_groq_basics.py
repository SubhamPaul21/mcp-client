from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "look at today's i.e. 6th August, 2025 price of Bitcoin and suggest if investing in it is right at this moment",
        }
    ],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None,
    tools=[{"type": "browser_search"}],
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
