from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

app = FastAPI(title="RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_validation(cls, query: str) -> str :
        if not query.strip():
            raise ValueError("Query cannot be empty")
        elif len(query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        elif len(query.strip()) > 500:
            raise ValueError("Query cannot exceed 500 characters")
        return query.strip()

bad_word_list = ["badword1", "badword2", "badword3"]  

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query")
def query_endpoint(request: QueryRequest):
    if request.query.lower() in bad_word_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The query '{request.query}' is not allowed."
        )
    return {"answer": f"You asked: {request.query}", "sources": []}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"}
    )

@app.get("/crash")
async def crash_endpoint():
    raise RuntimeError("Something went critically wrong inside the server!")

