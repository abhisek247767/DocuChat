# controllers/chat_controller.py
from fastapi import APIRouter
from models.chat_models import ChatRequest, ChatResponse
import services.chat_service as chat_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    answer = chat_service.chat(request.question)
    return ChatResponse(answer=answer)

@router.get("/")
def root():
    return {"message": "Chatbot API is running!"}
