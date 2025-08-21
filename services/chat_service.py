# services/chat_service.py
import os
from pathlib import Path

from server import setup_environment, load_documents, create_rag_pipeline, ask_chatbot

def init_chatbot():
    setup_environment()
    current_dir = Path(__file__).parent

    DATA_FILE_PATH = current_dir.parent / "data" / "dataset_childrenbook.pdf"
    DATA_FILE_TYPE = "pdf"

    documents = load_documents(DATA_FILE_PATH, DATA_FILE_TYPE)
    rag_chain = create_rag_pipeline(documents)
    return rag_chain

rag_chain = init_chatbot()

def chat(question: str) -> str:
    if not rag_chain:
        return "RAG chain not initialized."
    
    response = ask_chatbot(question, rag_chain)
    return response.get("answer", "No answer found")  # Extract the string answer