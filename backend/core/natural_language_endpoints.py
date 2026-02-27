"""
Natural Language Interface API Endpoints
Provides REST API for conversational AI capabilities
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import structlog

from .natural_language_interface import (
    get_natural_language_interface,
    QueryResponse,
    ConversationContext,
    IntentType
)
from .auth import get_current_user
from .models import User
from .exceptions import NLPProcessingError

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/nlp", tags=["Natural Language Processing"])


class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    query: str = Field(..., description="Natural language query from user")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    data_context: Optional[Dict[str, Any]] = Field(None, description="Additional data context")


class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    conversation_id: str
    response: QueryResponse
    context_updated: bool


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history"""
    conversation_id: str
    user_id: str
    session_history: List[Dict[str, Any]]
    current_focus: Optional[str]
    last_query_time: Optional[datetime]


class InsightRequest(BaseModel):
    """Request model for generating insights from data"""
    data: Dict[str, Any] = Field(..., description="Data to analyze")
    intent_type: IntentType = Field(..., description="Type of analysis to perform")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")


class VisualizationRequest(BaseModel):
    """Request model for creating visualizations"""
    data: Any = Field(..., description="Data for visualization")
    viz_type: str = Field(..., description="Type of visualization")
    title: Optional[str] = Field(None, description="Visualization title")
    options: Optional[Dict[str, Any]] = Field(None, description="Visualization options")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Process a natural language query and return AI response
    
    This endpoint handles conversational AI interactions, maintaining context
    across multiple exchanges and providing intelligent responses with
    visualizations and recommendations.
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Process the query
        response = await get_natural_language_interface().process_query(
            query=request.query,
            conversation_id=conversation_id,
            user_id=str(current_user.id),
            data_context=request.data_context
        )
        
        logger.info(
            "Chat query processed",
            user_id=current_user.id,
            conversation_id=conversation_id,
            query_length=len(request.query),
            response_confidence=response.confidence_score
        )
        
        return ChatResponse(
            conversation_id=conversation_id,
            response=response,
            context_updated=True
        )
        
    except NLPProcessingError as e:
        logger.error("NLP processing error", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=422, detail=f"NLP processing failed: {str(e)}")
    
    except Exception as e:
        logger.error("Chat processing error", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Internal server error during chat processing")


@router.get("/conversations/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve conversation history and context
    
    Returns the complete conversation history, current focus area,
    and context information for a specific conversation.
    """
    try:
        context = await get_natural_language_interface().context_manager.get_context(
            conversation_id, str(current_user.id)
        )
        
        return ConversationHistoryResponse(
            conversation_id=context.conversation_id,
            user_id=context.user_id,
            session_history=context.session_history,
            current_focus=context.current_focus,
            last_query_time=context.last_query_time
        )
        
    except Exception as e:
        logger.error("Error retrieving conversation history", 
                    conversation_id=conversation_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Clear conversation history and context
    
    Removes all conversation history and resets the context for
    the specified conversation ID.
    """
    try:
        if conversation_id in get_natural_language_interface().context_manager.active_contexts:
            del get_natural_language_interface().context_manager.active_contexts[conversation_id]
        
        logger.info("Conversation cleared", 
                   conversation_id=conversation_id, user_id=current_user.id)
        
        return {"message": "Conversation cleared successfully"}
        
    except Exception as e:
        logger.error("Error clearing conversation", 
                    conversation_id=conversation_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear conversation")


@router.post("/insights", response_model=QueryResponse)
async def generate_insights(
    request: InsightRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generate insights from data using AI analysis
    
    Analyzes provided data and generates insights, recommendations,
    and visualizations based on the specified intent type.
    """
    try:
        # Create intent object from request
        from .natural_language_interface import Intent, Entity
        
        intent = Intent(
            intent_type=request.intent_type,
            confidence=1.0,  # High confidence for direct API calls
            entities=[],
            parameters=request.parameters or {}
        )
        
        # Generate insights
        response = await get_natural_language_interface().generate_insights(
            data=request.data,
            query_intent=intent
        )
        
        logger.info(
            "Insights generated",
            user_id=current_user.id,
            intent_type=request.intent_type.value,
            data_size=len(str(request.data))
        )
        
        return response
        
    except Exception as e:
        logger.error("Error generating insights", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to generate insights")


@router.post("/visualizations")
async def create_visualization(
    request: VisualizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create visualization specification from data
    
    Generates chart configurations and visualization specifications
    that can be used by frontend charting libraries.
    """
    try:
        # Create visualization specification
        viz_spec = await get_natural_language_interface().create_visualization(
            data=request.data,
            viz_type=request.viz_type
        )
        
        # Add title and options if provided
        if request.title:
            viz_spec["title"] = request.title
        
        if request.options:
            viz_spec["options"] = {**viz_spec.get("options", {}), **request.options}
        
        logger.info(
            "Visualization created",
            user_id=current_user.id,
            viz_type=request.viz_type,
            has_title=bool(request.title)
        )
        
        return viz_spec
        
    except Exception as e:
        logger.error("Error creating visualization", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to create visualization")


@router.get("/conversations")
async def list_conversations(
    current_user: User = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=100, description="Number of conversations to return")
):
    """
    List active conversations for the current user
    
    Returns a list of active conversation IDs and their metadata
    for the authenticated user.
    """
    try:
        user_conversations = []
        
        for conversation_id, context in get_natural_language_interface().context_manager.active_contexts.items():
            if context.user_id == str(current_user.id):
                user_conversations.append({
                    "conversation_id": conversation_id,
                    "current_focus": context.current_focus,
                    "last_query_time": context.last_query_time,
                    "message_count": len(context.session_history)
                })
        
        # Sort by last query time (most recent first)
        user_conversations.sort(
            key=lambda x: x["last_query_time"] or datetime.min,
            reverse=True
        )
        
        return {
            "conversations": user_conversations[:limit],
            "total_count": len(user_conversations)
        }
        
    except Exception as e:
        logger.error("Error listing conversations", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to list conversations")


@router.post("/conversations/{conversation_id}/context")
async def update_conversation_context(
    conversation_id: str,
    context_update: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Update conversation context with additional information
    
    Allows manual updates to conversation context, such as setting
    user preferences or adding external context information.
    """
    try:
        context = await get_natural_language_interface().context_manager.get_context(
            conversation_id, str(current_user.id)
        )
        
        # Update user preferences
        if "user_preferences" in context_update:
            context.user_preferences.update(context_update["user_preferences"])
        
        # Update current focus
        if "current_focus" in context_update:
            context.current_focus = context_update["current_focus"]
        
        # Add custom context data
        if "custom_data" in context_update:
            if not hasattr(context, "custom_data"):
                context.custom_data = {}
            context.custom_data.update(context_update["custom_data"])
        
        logger.info(
            "Conversation context updated",
            conversation_id=conversation_id,
            user_id=current_user.id,
            update_keys=list(context_update.keys())
        )
        
        return {"message": "Context updated successfully"}
        
    except Exception as e:
        logger.error("Error updating conversation context", 
                    conversation_id=conversation_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update conversation context")


@router.get("/performance")
async def get_nlp_performance_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get NLP performance statistics
    
    Returns performance metrics including cache hit rates,
    active conversations, and model loading status.
    """
    try:
        stats = await get_natural_language_interface().get_performance_stats()
        
        return {
            "status": "healthy",
            "performance_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Error getting NLP performance stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get performance statistics")


@router.post("/clear-cache")
async def clear_nlp_caches(
    current_user: User = Depends(get_current_user)
):
    """
    Clear NLP caches to free memory
    
    Clears intent classification caches and old conversation contexts
    to improve performance and free memory.
    """
    try:
        await get_natural_language_interface().clear_caches()
        
        logger.info("NLP caches cleared", user_id=current_user.id)
        
        return {"message": "NLP caches cleared successfully"}
        
    except Exception as e:
        logger.error("Error clearing NLP caches", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear caches")


@router.get("/health")
async def nlp_health_check():
    """
    Health check endpoint for NLP services
    
    Returns the status of NLP models and services to ensure
    they are loaded and functioning correctly.
    """
    try:
        # Check if models are loaded
        nlp_interface = get_natural_language_interface()
        parser_status = hasattr(nlp_interface.query_parser, 'intent_model')
        generator_status = hasattr(nlp_interface.response_generator, 'openai_client')
        context_status = len(nlp_interface.context_manager.active_contexts) >= 0
        
        status = {
            "status": "healthy" if all([parser_status, generator_status, context_status]) else "degraded",
            "components": {
                "query_parser": "healthy" if parser_status else "error",
                "response_generator": "healthy" if generator_status else "error",
                "context_manager": "healthy" if context_status else "error"
            },
            "active_conversations": len(get_natural_language_interface().context_manager.active_contexts),
            "timestamp": datetime.now().isoformat()
        }
        
        return status
        
    except Exception as e:
        logger.error("NLP health check failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/conversations/cleanup")
async def cleanup_old_conversations(
    max_age_hours: int = Query(24, ge=1, le=168, description="Maximum age in hours"),
    current_user: User = Depends(get_current_user)
):
    """
    Clean up old conversation contexts
    
    Removes conversation contexts that are older than the specified
    age to free up memory and maintain performance.
    """
    try:
        initial_count = len(get_natural_language_interface().context_manager.active_contexts)
        
        await get_natural_language_interface().context_manager.cleanup_old_contexts(max_age_hours)
        
        final_count = len(get_natural_language_interface().context_manager.active_contexts)
        cleaned_count = initial_count - final_count
        
        logger.info(
            "Conversation cleanup completed",
            user_id=current_user.id,
            cleaned_count=cleaned_count,
            remaining_count=final_count,
            max_age_hours=max_age_hours
        )
        
        return {
            "message": f"Cleaned up {cleaned_count} old conversations",
            "cleaned_count": cleaned_count,
            "remaining_count": final_count
        }
        
    except Exception as e:
        logger.error("Error during conversation cleanup", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cleanup conversations")