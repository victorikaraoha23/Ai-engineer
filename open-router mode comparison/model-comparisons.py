import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url="https://openrouter.ai/api/v1")
response = client.chat.completions.create(
    model="openouter/free", message=[{"role": "user", "content": "define noun"}]
)  

print(response.choices[0].message.content)
