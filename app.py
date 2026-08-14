from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str


@app.get("/health/{status}")
def health_check(status: int):
    return {"status": status}


@app.post("/query")
def query_endpoint(request: QueryRequest):
    return {"answer": f"You asked: {request.query}", "sources": []}