from fastapi import APIRouter, HTTPException

from app.schema.policy_chatbot_schema import (
    PolicyChatbotAskRequest,
    PolicyChatbotAskResponse,
)
from app.service.policy_chatbot.chat_app import answer_policy_chatbot_query

router = APIRouter()

@router.post("/ask", response_model=PolicyChatbotAskResponse)
def ask_policy_chatbot(request: PolicyChatbotAskRequest):
    try:
        result = answer_policy_chatbot_query(request)
        return PolicyChatbotAskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))