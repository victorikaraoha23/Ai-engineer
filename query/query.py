import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openaiclient = OpenAI(base_url="https://openrouter.ai/api/v1")

chromaclient = chromadb.PersistentClient(path="./chroma_db")
collection = chromaclient.get_or_create_collection(name="python_bible")

def embed_text(text, model="nvidia/llama-nemotron-embed-vl-1b-v2:free"):
    response = openaiclient.embeddings.create(
        model=model, input=text, encoding_format="float"
    )
    return response.data[0].embedding


query = sys.argv[1] if len(sys.argv) > 1 else "What is a Python dictionary?"  
query_embedding = embed_text(query)

results = collection.query(query_embeddings=[query_embedding], n_results=1)
context_chunks = []
for i, doc in enumerate(results["documents"][0]):
    context_chunks.append(f"[Chunk {i + 1}]\n{doc}")

context = "\n\n".join(context_chunks)

response = openaiclient.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an assistant that answers questions only from "
                "provided context. If the context doesn't contain the answer, "
                "say 'I don't know.' Cite which chunks you used."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ],max_tokens=100,
)

# Extract and print the assistant's text response
print(response.choices[0].message.content)
print("\nSources:")
for i, meta in enumerate(results["metadatas"][0]):
    print(f"- {meta['source']} (chunk {meta['chunk_index']})")
