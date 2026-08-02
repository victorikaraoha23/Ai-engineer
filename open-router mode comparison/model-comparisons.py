import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url="https://openrouter.ai/api/v1")

def openrouter_call():
    model_list=["openrouter/free","~deepseek/deepseek-v4-flash-latest"]
    for models in model_list:
        response = client.chat.completions.create(
            model=models,
            messages=[{"role": "user", "content": sys.argv[1]}],
        )
        return response.choices[0].message.content


print(openrouter_call())
