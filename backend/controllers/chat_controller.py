import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.schemas.chat_schema import ChatRequest, ChatResponse
from backend.schemas.predict_shema import PredictionResponse
from backend.services.leaf_analysis_service import analyze_leaf, format_for_agent
from backend.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])
rag_service = RAGService()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_IMAGE_QUESTION = "What is wrong with this tomato leaf, and what product can I use?"


class ChatTextResponse(ChatResponse):
    conversation_id: str


class ChatImageResponse(ChatResponse):
    # Full prediction + product records are returned for the UI.
    analysis: Optional[PredictionResponse] = None
    conversation_id: str


@router.post("", response_model=ChatTextResponse)
def chat(
    request: ChatRequest,
    conversation_id: str = Query(default=""),
) -> ChatTextResponse:
    active_id = conversation_id.strip() or uuid.uuid4().hex
    try:
        result = rag_service.ask(request.question, conversation_id=active_id)
        return ChatTextResponse(
            success=result["success"],
            answer=result["answer"],
            reason=result.get("reason"),
            conversation_id=active_id,
        )
    except Exception:
        logger.exception("Chat request failed")
        return ChatTextResponse(
            success=False,
            answer="",
            reason="The assistant failed to process the request.",
            conversation_id=active_id,
        )


@router.post("/image", response_model=ChatImageResponse)
def chat_with_image(
    question: str = Form(default=""),
    conversation_id: str = Form(default=""),
    file: UploadFile = File(..., description="Leaf image (jpg/png/webp)"),
) -> ChatImageResponse:
    active_id = conversation_id.strip() or uuid.uuid4().hex
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only jpg, png or webp images are allowed")

    try:
        analysis = analyze_leaf(file.file)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.exception("Leaf analysis failed")
        raise HTTPException(status_code=500, detail="Could not analyse the image") from error

    try:
        result = rag_service.ask(
            question.strip() or DEFAULT_IMAGE_QUESTION,
            context=format_for_agent(analysis),
            conversation_id=active_id,
        )
        return ChatImageResponse(
            success=result["success"],
            answer=result["answer"],
            reason=result.get("reason"),
            analysis=analysis,
            conversation_id=active_id,
        )
    except Exception as error:
        logger.exception("Chat agent failed after leaf analysis")
        return ChatImageResponse(
            success=False,
            answer="The leaf analysis completed, but the assistant could not write the response.",
            reason="agent_failed_after_leaf_analysis",
            analysis=analysis,
            conversation_id=active_id,
        )
