import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

app = FastAPI(title="RAG API", version="1.0.0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

request_timestamps: dict[str, list[float]] = {}
RATE_LIMIT = 10
RATE_LIMIT_WINDOW = 60.0


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    request_times = request_timestamps.setdefault(client_ip, [])
    current_time = time.monotonic()
    request_times[:] = [
        timestamp
        for timestamp in request_times
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]

    if len(request_times) >= RATE_LIMIT:
        latency = time.perf_counter() - start_time
        logger.info("%s %s 429 %.4fs", request.method, request.url.path, latency)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Try again later."},
            headers={"X-Process-Time": str(latency)},
        )

    request_times.append(current_time)
    response = await call_next(request)
    latency = time.perf_counter() - start_time
    logger.info(
        "%s %s %s %.4fs",
        request.method,
        request.url.path,
        response.status_code,
        latency,
    )
    response.headers["X-Process-Time"] = str(latency)
    return response


class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_validation(cls, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        elif len(query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        elif len(query.strip()) > 500:
            raise ValueError("Query cannot exceed 500 characters")
        return query.strip()


bad_word_list = ["badword1", "badword2", "badword3"]


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if request.query.lower() in bad_word_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The query '{request.query}' is not allowed.",
        )
    return {"answer": f"You asked: {request.query}", "sources": []}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )


@app.get("/crash")
async def crash_endpoint():
    raise RuntimeError("Something went critically wrong inside the server!")


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    if request.query.lower() in bad_word_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The query '{request.query}' is not allowed.",
        )

    async def stream_response():
        yield f"You asked: {request.query}\n"
        yield "Streaming response...\n"
        for i in range(5):
            yield f"Chunk {i + 1}\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(stream_response(), media_type="text/plain")
