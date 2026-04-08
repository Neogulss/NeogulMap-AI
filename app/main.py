from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.policy_chatbot_api import router as policy_chatbot_router
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.service.policy_chatbot.chat_app import initialize_policy_chatbot_resources


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    initialize_policy_chatbot_resources()
    yield
    
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    policy_chatbot_router,
    prefix="/api/policy-chatbot",
    tags=["policy-chatbot"],
)

@app.get("/health")
def health_check():
    return{
        "status": "ok",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }