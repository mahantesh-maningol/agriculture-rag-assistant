from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from backend.controllers.chat_controller import router as chat_router
from backend.controllers.predict_controller import router as predict_router
import logging

logging.basicConfig(level=logging.INFO)
from backend.database import (
    open_database_pool,
    close_database_pool,
    check_database_connection,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    open_database_pool()

    try:
        yield
    finally:
        close_database_pool()

app = FastAPI(
    title="Agriculture RAG API",
    version="1.0.0",
    description="API for Agriculture RAG system",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.get("/health/database")
def database_health():
    try:
        connected = check_database_connection()

        if not connected:
            raise HTTPException(
                status_code=503,
                detail="Database unavailable",
            )

        return {"database": "connected"}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )


app.include_router(chat_router)
app.include_router(predict_router)