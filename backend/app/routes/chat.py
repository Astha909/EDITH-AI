from fastapi import APIRouter

from app.models.chat_models import ChatRequest, ChatResponse
from app.services.ollama_service import generate_response

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    ai_response = generate_response(request.message)

    return ChatResponse(response=ai_response)