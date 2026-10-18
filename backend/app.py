from fastapi import FastAPI
from backend.controllers.chat_controller import router as chat_router
from backend.controllers.predict_controller import router as predict_router


app = FastAPI(
    title="Agriculture RAG API",
    version="1.0.0",
    description="API for Agriculture RAG system"
)

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

app.include_router(chat_router)
app.include_router(predict_router)