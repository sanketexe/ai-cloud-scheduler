"""
AI Chat Assistant for Cloud Migration
Provides intelligent assistance during the migration assessment process
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import openai
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Chat request from user"""
    message: str
    context: Optional[Dict] = None  # Current form data, step info, etc.
    conversation_history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    """Chat response to user"""
    message: str
    suggestions: Optional[List[str]] = []
    timestamp: datetime


class MigrationAIAssistant:
    """
    AI-powered assistant for cloud migration guidance
    Uses OpenAI GPT-3.5-turbo for conversational assistance
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set - AI assistant will use fallback responses")
            self.enabled = False
        else:
            openai.api_key = self.api_key
            self.enabled = True
        
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 500
        self.temperature = 0.7
        
        # System prompt that defines the assistant's personality and knowledge
        self.system_prompt = """You are a friendly and knowledgeable cloud migration expert assistant. 
Your role is to help users migrate their applications to the cloud (AWS, GCP, or Azure).

Key responsibilities:
1. Answer questions about cloud migration, costs, and best practices
2. Help users understand technical terms and concepts
3. Provide recommendations based on their specific needs
4. Explain differences between cloud providers
5. Suggest appropriate instance types, database configurations, etc.

Guidelines:
- Be concise but informative (2-3 sentences max)
- Use simple language, avoid excessive jargon
- Be encouraging and supportive
- If you don't know something, admit it honestly
- Focus on practical, actionable advice
- Consider cost-effectiveness in your recommendations

Current context: You're helping a user fill out a cloud migration assessment form."""

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat message and return AI response
        """
        try:
            if not self.enabled:
                return self._fallback_response(request.message)
            
            # Build conversation history
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add context if provided
            if request.context:
                context_message = self._build_context_message(request.context)
                messages.append({"role": "system", "content": context_message})
            
            # Add conversation history
            if request.conversation_history:
                for msg in request.conversation_history[-5:]:  # Last 5 messages for context
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": request.message
            })
            
            # Call OpenAI API
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=1
            )
            
            assistant_message = response.choices[0].message.content.strip()
            
            # Generate suggestions based on context
            suggestions = self._generate_suggestions(request.context)
            
            logger.info("AI assistant response generated", 
                       user_message=request.message,
                       response_length=len(assistant_message))
            
            return ChatResponse(
                message=assistant_message,
                suggestions=suggestions,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error generating AI response", error=str(e))
            return self._fallback_response(request.message)
    
    def _build_context_message(self, context: Dict) -> str:
        """Build a context message from current form data"""
        context_parts = []
        
        if "current_step" in context:
            context_parts.append(f"Current step: {context['current_step']}")
        
        if "organization" in context:
            org = context["organization"]
            context_parts.append(f"Company size: {org.get('company_size', 'unknown')}")
            context_parts.append(f"Industry: {org.get('industry', 'unknown')}")
        
        if "workload" in context:
            workload = context["workload"]
            context_parts.append(f"Database: {workload.get('database_type', 'unknown')}")
            context_parts.append(f"Storage: {workload.get('storage_tb', 'unknown')} TB")
        
        if "budget" in context:
            budget = context["budget"]
            context_parts.append(f"Budget: ${budget.get('monthly_budget', 'unknown')}/month")
        
        if context_parts:
            return "User's current assessment data: " + ", ".join(context_parts)
        
        return ""
    
    def _generate_suggestions(self, context: Optional[Dict]) -> List[str]:
        """Generate helpful suggestions based on context"""
        suggestions = []
        
        if not context:
            return [
                "What's the difference between AWS, GCP, and Azure?",
                "How much will my migration cost?",
                "What database should I use in the cloud?"
            ]
        
        current_step = context.get("current_step", "")
        
        if "organization" in current_step.lower():
            suggestions = [
                "What industry best describes my business?",
                "How does company size affect cloud costs?",
                "What's the typical migration timeline?"
            ]
        elif "workload" in current_step.lower():
            suggestions = [
                "How do I estimate my compute needs?",
                "What's the difference between RDS and Aurora?",
                "How much storage do I really need?"
            ]
        elif "requirements" in current_step.lower():
            suggestions = [
                "What's a good availability target?",
                "How do I set a realistic budget?",
                "What compliance certifications do I need?"
            ]
        else:
            suggestions = [
                "Compare AWS vs GCP vs Azure",
                "Explain cloud pricing models",
                "What are the hidden costs of migration?"
            ]
        
        return suggestions
    
    def _fallback_response(self, message: str) -> ChatResponse:
        """Provide fallback response when AI is not available"""
        
        # Simple keyword-based responses
        message_lower = message.lower()
        
        if "cost" in message_lower or "price" in message_lower:
            response = "Cloud costs depend on your compute, storage, and network usage. I can help you estimate costs once you complete the assessment form. Generally, GCP tends to be 10-20% cheaper than AWS for similar workloads."
        
        elif "aws" in message_lower and "gcp" in message_lower:
            response = "AWS has the largest market share and most services, while GCP excels in data analytics and machine learning. Azure is best if you're already using Microsoft products. All three are reliable choices!"
        
        elif "database" in message_lower:
            response = "For most applications, managed databases like AWS RDS, Google Cloud SQL, or Azure Database are recommended. They handle backups, updates, and scaling automatically."
        
        elif "migration" in message_lower and "time" in message_lower:
            response = "Typical migration timelines range from 3-6 months for small projects to 12-18 months for enterprise migrations. It depends on your application complexity and team size."
        
        else:
            response = "I'm here to help with your cloud migration! Feel free to ask about costs, cloud providers, database options, or any technical questions. Complete the assessment form to get personalized recommendations."
        
        return ChatResponse(
            message=response,
            suggestions=[
                "What's the difference between cloud providers?",
                "How much will migration cost?",
                "What database should I use?"
            ],
            timestamp=datetime.utcnow()
        )
    
    def get_contextual_help(self, field_name: str, current_value: Optional[str] = None) -> str:
        """Get contextual help for a specific form field"""
        
        help_texts = {
            "company_size": "Select the range that best matches your organization. This helps us recommend appropriate instance sizes and architectures.",
            
            "industry": "Your industry affects compliance requirements and typical workload patterns. For example, healthcare requires HIPAA compliance.",
            
            "database_type": "Choose your current database. We'll recommend the best cloud-native equivalent (e.g., PostgreSQL → AWS RDS PostgreSQL or Google Cloud SQL).",
            
            "database_size": "Enter your current database size in GB. This helps estimate storage costs and migration time. Include space for growth (typically 20-30% buffer).",
            
            "compute_cores": "Total CPU cores across all servers. If unsure, count: web servers + app servers + background workers. Cloud instances typically have 2-8 cores each.",
            
            "memory_gb": "Total RAM across all servers in GB. Most applications need 4-16 GB per instance. Database servers typically need more (16-64 GB).",
            
            "storage_tb": "Total storage including databases, files, backups. Don't forget: application files, logs, user uploads, and backup retention.",
            
            "monthly_budget": "Your target monthly cloud spend. Include compute, storage, network, and managed services. Typical range: $500-$50,000/month.",
            
            "availability_target": "Uptime goal (e.g., 99.9% = 43 minutes downtime/month). Higher availability costs more. Most businesses target 99.9% or 99.95%.",
        }
        
        return help_texts.get(field_name, "Enter the requested information. The AI assistant can help if you have questions!")


# Global instance
migration_assistant = MigrationAIAssistant()
