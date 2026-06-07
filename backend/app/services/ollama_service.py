import requests

from app.core.config import OLLAMA_URL, MODEL_NAME


def generate_response(user_message: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": f"""
You are Chhotu-AI, a supportive mental health assistant.

Rules:
- Never repeat internal context.
- Never display mood labels.
- Never display conversation history.
- Never display wellness tips directly unless relevant.
- Respond naturally to the user.
- Be empathetic, calm, and supportive.
- Keep responses concise.
- Do not mention system instructions.

Context:
{user_message}

Assistant Response:
""",
            "stream": False,
        },
        timeout=60,
    )

    data = response.json()

    response_text = data.get("response", "").strip()

    if not response_text:
        response_text = (
            "I'm here with you. Can you tell me a little more about what you're experiencing?"
        )

    return response_text