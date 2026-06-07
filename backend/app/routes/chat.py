from fastapi import APIRouter
import time

from app.models.chat_models import ChatRequest, ChatResponse
from app.services.ollama_service import generate_response

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    start_time = time.time()

    print("\n========== NEW REQUEST ==========")
    print("USER:", request.message[:200])

    ai_response = generate_response(
        request.message
    )

    duration = round(
        time.time() - start_time,
        2,
    )

    print("AI:", ai_response[:300])
    print("TIME:", duration, "seconds")
    print("=================================\n")

    return ChatResponse(
        response=ai_response
    )