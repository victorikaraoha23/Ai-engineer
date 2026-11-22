# AI Engineering Learning Lab

A small Python learning repository exploring the building blocks of retrieval-augmented generation (RAG): document extraction, token-based chunking, embedding generation, vector storage, semantic retrieval, and context-grounded language-model responses.

## Current prototype

The repository contains a proof-of-concept pipeline that:

1. Extracts text from the included Python reference PDF using PyMuPDF.
2. Tokenizes and splits the text into overlapping chunks with `tiktoken`.
3. Requests embeddings through an OpenAI-compatible endpoint configured for OpenRouter.
4. Stores document chunks and vectors in a persistent ChromaDB collection.
5. Retrieves a matching chunk for a query.
6. Sends the retrieved context and question to a chat model with instructions to answer from the supplied context.

The separate FastAPI application in `app.py` is an API exercise with request validation, basic in-memory rate limiting, health checks, JSON error handling, and a demonstration streaming endpoint. Its current query endpoint is a simple placeholder response; it is not yet connected to the RAG pipeline in `rag/rag.py`.

## Technology

- Python 3.13+
- FastAPI and Pydantic
- ChromaDB
- PyMuPDF
- tiktoken
- OpenAI-compatible SDK configured for OpenRouter
- uv for dependency management

## Setup

Requirements: Python version matching `.python-version` and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/victorikaraoha23/Ai-engineer.git
cd Ai-engineer
uv sync
```

The embedding and chat scripts expect an OpenRouter API key in the environment. Configure it locally; do not commit credentials. The vector database is persisted in `chroma_db/`.

## Running the experiments

Build the local vector collection from the included PDF:

```bash
uv run python embed.py
```

Query the RAG pipeline:

```bash
uv run python rag/rag.py "What is a Python dictionary?"
```

Start the FastAPI exercise:

```bash
uv run uvicorn app:app --reload
```

## Important limitations

- This is a learning prototype, not a hosted or production RAG service.
- The FastAPI query route is not wired to the vector retrieval pipeline yet.
- The current retrieval example requests one matching chunk; it is not a complete retrieval/reranking or citation system.
- The included ChromaDB directory is a local generated data store. Rebuild it with `embed.py` if the environment or embedding model changes.
- Model availability, API limits, and embedding compatibility depend on the configured provider.

## Learning goals

This repository records practical exploration of Python APIs, embeddings, vector databases, document processing, and context-grounded LLM applications. It should be treated as an experiment log rather than a finished product.
