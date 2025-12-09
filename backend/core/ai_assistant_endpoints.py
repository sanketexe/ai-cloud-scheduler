"""
API endpoints for AI Chat Assistant
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import structlog

from .ai_assistant import (
    migration_assistant,
    ChatRequest,
    ChatResponse
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/migration/assistant", tags=["AI Assistant"])


@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    """
    Chat with the AI migration assistant
    
    The assistant provides:
    - Answers to migration questions
    - Contextual help based on current form data
    - Suggestions for next steps
    - Best practices and recommendations
    """
    try:
        response = await migration_assistant.chat(request)
        return response
    
    except Exception as e:
        logger.error("Chat endpoint error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat message"
        )


@router.get("/help/{field_name}")
async def get_field_help(field_name: str, current_value: Optional[str] = None):
    """
    Get contextual help for a specific form field
    """
    try:
        help_text = migration_assistant.get_contextual_help(field_name, current_value)
        return {
            "field_name": field_name,
            "help_text": help_text
        }
    
    except Exception as e:
        logger.error("Field help error", error=str(e), field=field_name)
        raise HTTPException(
            status_code=500,
            detail="Failed to get field help"
        )


@router.get("/suggestions")
async def get_suggestions(context: Optional[str] = None):
    """
    Get suggested questions based on current context
    """
    try:
        context_dict = {"current_step": context} if context else None
        suggestions = migration_assistant._generate_suggestions(context_dict)
        return {
            "suggestions": suggestions
        }
    
    except Exception as e:
        logger.error("Suggestions error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to generate suggestions"
        )


@router.get("/status")
async def get_assistant_status():
    """
    Check if AI assistant is enabled and working
    """
    return {
        "enabled": migration_assistant.enabled,
        "model": migration_assistant.model if migration_assistant.enabled else None,
        "status": "ready" if migration_assistant.enabled else "fallback_mode"
    }
