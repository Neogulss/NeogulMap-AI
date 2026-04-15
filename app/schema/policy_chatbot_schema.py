from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class UserProfileInput(BaseModel):
    industry: Optional[str] = Field(default=None, description="업종")
    age: Optional[int] = Field(default=None, ge=0, le=120, description="나이")
    has_business_registration: Optional[bool] = Field(
        default=None,
        description="사업자등록 여부"
    )
    business_stage: Optional[Literal["예비창업", "창업", "재창업"]] = Field(
        default=None,
        description="사업여부(예비창업/창업/재창업)"
    )
    region: Optional[str] = Field(default=None, description="지역")
    startup_status: Optional[str] = Field(default=None, description="창업 상태")
    
    
class PolicyChatbotAskRequest(BaseModel):
    user_query: str = Field(..., min_length=1, description="사용자 질문")
    user_profile: Optional[UserProfileInput] = None
    session_idx: Optional[int] = None
    
class RetrievedDocumentItem(BaseModel):
    source: str
    chunk_text: str
    faiss_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None

class LatencyItem(BaseModel):
    retrieval_ms: int
    llm_generation_ms: int
    turn_latency_ms: int
    
class ChatLogPayload(BaseModel):
    user_query: str
    bot_response: str
    model: str
    turn_latency_ms: int

class RagSystemLatency(BaseModel):
    retrieval_ms: int
    llm_generation_ms: int
    
class RagLogPayload(BaseModel):
    retrieved_documents: List[RetrievedDocumentItem]
    system_latency: RagSystemLatency
    
class PolicyChatbotAskResponse(BaseModel):
    type: Literal["policy_or_loan", "service_intro", "unsupported", "error"]
    answer: str
    model: str
    session_title_suggestion: str
    retrieved_documents: List[RetrievedDocumentItem]
    latency: LatencyItem
    chat_log_payload: ChatLogPayload
    rag_log_payload: RagLogPayload
