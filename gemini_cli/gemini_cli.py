import sys

from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()

client = genai.Client()


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def gemini_call():
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=sys.argv[1]
    )
    # Gemini 2.5 Flash Pricing (per 1,000,000 tokens)
    input_rate = 0.30 / 1_000_000
    output_rate = 2.50 / 1_000_000

    # Calculate costs
    input_cost = response.usage_metadata.prompt_token_count * input_rate
    output_cost = response.usage_metadata.candidates_token_count * output_rate
    total_cost = input_cost + output_cost

    return (
        f"response: {response.text}\n"
        f"input_cost: ${input_cost:.6f}\n"
        f"output_cost: ${output_cost:.6f}\n"
        f"total_cost: ${total_cost:.6f}\n"
        f"usage.thoughts_token_count: {response.usage_metadata.thoughts_token_count}\n"
        f"usage.candidates_token_count: {response.usage_metadata.candidates_token_count}\n"
        f"usage.prompt_token_count: {response.usage_metadata.prompt_token_count}\n"
        f"usage.total_token_count: {response.usage_metadata.total_token_count}\n"
    )


print(gemini_call())
