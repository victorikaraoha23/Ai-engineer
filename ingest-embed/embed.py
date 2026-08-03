import chromadb
import pymupdf
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = chromadb.Client()
collection= client.create_collection(name="python bible")``

client = OpenAI(base_url="https://openrouter.ai/api/v1")

with pymupdf.open("the-python-bible.pdf") as doc:
    full_text = ""
    for page in doc:
        full_text += str(page.get_text())
encoding = tiktoken.get_encoding("cl100k_base")
token_id = encoding.encode(full_text)
print(token_id)
print(f"Number of tokens in full text: {len(token_id)}")


def chunk_tokens(tokens, chunk_size=500, overlap=50):
    if not tokens:
        return []
    chunks = []
    start = 0
    text_length = len(tokens)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = tokens[start:end]
        decoded_chunk = encoding.decode(chunk)
        if chunk:
            chunks.append(decoded_chunk)
        if end == text_length:
            break
        start = end - overlap
    return chunks


print(chunk_tokens(token_id, chunk_size=500, overlap=50))
print(f"Number of chunks: {len(chunk_tokens(token_id, chunk_size=500, overlap=50))}")


def embed_text(text, model="openai/text-embedding-3-small"):
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


print(len(embed_text(chunk_tokens(token_id, chunk_size=500, overlap=50)[0])))
