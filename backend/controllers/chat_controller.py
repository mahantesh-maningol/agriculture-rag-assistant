import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.schemas.chat_schema import ChatRequest, ChatResponse
from backend.schemas.predict_shema import PredictionResponse
from backend.services.leaf_analysis_service import analyze_leaf, format_for_agent
from backend.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

rag_service = RAGService()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_IMAGE_QUESTION = "What is wrong with this tomato leaf, and what product can I use?"


class ChatImageResponse(ChatResponse):
    # Full prediction + product records, so the client can render product cards
    # even if the written answer is short or the LLM call fails.
    analysis: Optional[PredictionResponse] = None


# Plain `def`: the RAG agent call is blocking, so run it in a worker thread.
@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = rag_service.ask(request.question)
        return ChatResponse(success=result['success'], answer=result['answer'], reason=result.get("reason"))
    except Exception as e:
        return ChatResponse(success=False, answer="", reason=str(e))


# multipart/form-data: question (optional text) + file (leaf image)
@router.post("/image", response_model=ChatImageResponse)
def chat_with_image(
    question: str = Form(""),
    file: UploadFile = File(..., description="Leaf image (jpg/png/webp)"),
) -> ChatImageResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only jpg, png or webp images are allowed")

    try:
        analysis = analyze_leaf(file.file)
    except ValueError as e:                       # not a valid image
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Leaf analysis failed")
        raise HTTPException(status_code=500, detail="Could not analyse the image")

    try:
        result = rag_service.ask(
            question.strip() or DEFAULT_IMAGE_QUESTION,
            context=format_for_agent(analysis),
        )
        return ChatImageResponse(
            success=result["success"],
            answer=result["answer"],
            reason=result.get("reason"),
            analysis=analysis,
        )
    except Exception:
        logger.exception("Chat agent failed after leaf analysis")
        return ChatImageResponse(
            success=False,
            answer="",
            reason="The assistant could not write an answer, but the image analysis is included.",
            analysis=analysis,
        )