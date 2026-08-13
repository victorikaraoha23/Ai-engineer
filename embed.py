import chromadb
import pymupdf
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

chromaclient = chromadb.PersistentClient(path="./chroma_db")
openaiclient = OpenAI(base_url="https://openrouter.ai/api/v1")

with pymupdf.open("the-python-bible.pdf") as doc:
    full_text = ""
    for page in doc:
        full_text += str(page.get_text())
encoding = tiktoken.get_encoding("cl100k_base")
token_id = encoding.encode(full_text)


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


print(f"Number of chunks: {len(chunk_tokens(token_id, chunk_size=500, overlap=50))}")


def embed_text(text, model="nvidia/llama-nemotron-embed-vl-1b-v2:free"):
    response = openaiclient.embeddings.create(
        model=model, input=text, encoding_format="float"
    )
    return response.data[0].embedding


collection = chromaclient.get_or_create_collection(name="python_bible")
chunks = chunk_tokens(token_id, chunk_size=500, overlap=50)
embeddings = []

for chunk in chunks:
    embedding = embed_text(chunk)
    embeddings.append(embedding)

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[{"source": "the-python-bible.pdf", "chunk_index": i}],
        ids=[f"chunk_{i}"],
    )

print(f"Stored {collection.count()} chunks in chromadb")

query = "What is a Python dictionary?"
query_embedding = embed_text(query)

results = collection.query(query_embeddings=[query_embedding], n_results=1)

print(f"\nQuery: {query}")
print("Top 1 chunk:")
for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- Chunk {i + 1} ---")
    print(doc[:300])
