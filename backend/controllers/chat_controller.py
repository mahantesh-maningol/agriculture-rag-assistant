from fastapi import APIRouter
from backend.schemas.chat_schema import ChatRequest, ChatResponse
from backend.services.rag_service import RAGService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

rag_service = RAGService()

@router.get("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = rag_service.ask(request.question)
        return ChatResponse(success=result['success'], answer=result['answer'], reason=result.get("reason"))
    except Exception as e:
        return ChatResponse(success=False, answer="", reason=str(e))
