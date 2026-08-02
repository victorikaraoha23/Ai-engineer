import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url="https://openrouter.ai/api/v1")


def openrouter_call():
    # cost per 1,000,000 tokens for each model
    model_costs_input = {
        "tencent/hy3-preview": 0.063,
        "~deepseek/deepseek-v4-flash-latest": 0.09,
        "google/gemini-2.5-flash-lite": 0.10,
    }
    model_costs_output = {
        "tencent/hy3-preview": 0.21,
        "~deepseek/deepseek-v4-flash-latest": 0.18,
        "google/gemini-2.5-flash-lite": 0.40,
    }
    model_list = [
        "tencent/hy3-preview",
        "~deepseek/deepseek-v4-flash-latest",
        "google/gemini-2.5-flash-lite",
    ]
    
    
    for models in model_list:
        try:
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model=models,
                messages=[
                    {
                        "role": "user",
                        "content": sys.argv[1],
                    }
                ],
            )
            end_time = time.perf_counter()
            print(f"Time taken for {models}: {end_time - start_time:.2f} seconds")
            print(f"Total tokens for {models}: {response.usage.total_tokens}")
            print(
                f"Completion tokens (output) for {models}: {response.usage.completion_tokens}"
            )
            print(f"Prompt tokens (input) for {models}: {response.usage.prompt_tokens}")
            print(
                f"cost for {models}: ${(response.usage.prompt_tokens * model_costs_input[models] + response.usage.completion_tokens * model_costs_output[models]) / 1_000_000:.6f}\n"
            )
            
        except Exception as e:
            print(f"An error occurred: {e}")
    print("=====================summary===================")
    print("fastest model:")
    print("cheapest model:")


openrouter_call()
