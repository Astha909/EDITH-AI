from fastapi import FastAPI

from app.core.config import APP_NAME
from app.routes.chat import router as chat_router

app = FastAPI(title=APP_NAME)

app.include_router(chat_router)