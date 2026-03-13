"""
AI Chat Assistant for Cloud Migration
Provides intelligent assistance during the migration assessment process
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import openai
from pydantic import BaseModel
import json
import structlog
from backend.app.api.scaling_rules_endpoints import get_engine

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


class PlatformAIAssistant:
    """
    AI-powered assistant for the entire FinOps platform.
    Can answer questions about costs, migration, and verify real-time AWS data.
    """
    
    def __init__(self, aws_service=None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.aws_service = aws_service
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set - AI assistant will use fallback responses")
            self.enabled = False
        else:
            openai.api_key = self.api_key
            self.enabled = True
        
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 500
        self.temperature = 0.7
        
        # System prompt for the full platform
        self.system_prompt = """You are the 'CloudPilot' AI assistant for a sophisticated Cloud FinOps and Migration platform.
Your role is to help users manage their AWS infrastructure, optimize costs, and plan migrations.

Capabilities:
1. Real-time Verification: You can see real-time data about the user's AWS account (EC2 counts, S3 buckets, current costs).
2. Cost Optimization: Explain RDS scheduling, rightsizing recommendations, and anomaly detection.
3. Migration Planning: Guide users through on-premises to cloud TCO analysis.
6. Compliance: Explain security findings and tagging policies.

**NEW ABILITY: Auto-Scaling Management**
You have the ability to manage Auto-Scaling rules directly for the user! 
- If the user asks to "create a rule" or "automate scaling when CPU is high", call `create_auto_scaling_rule`.
- If the user asks to "scale up my EBS right now" or "increase storage for vol-123", call `execute_scaling_action`. 
- ALWAY ask for missing mandatory details before calling the function (e.g., "Which resource ID should I scale?").

Guidelines:
- Be professional, concise, and technical but accessible.
- If the user asks for counts (e.g., 'How many instances?'), check the provided context data.
- If data is missing or AWS is not connected, politely inform the user.
- Always prioritize cost-saving advice.
"""

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat message with platform-wide context and real-time data."""
        try:
            # 1. Detect if real-time data is needed based on query
            real_time_context = self._get_real_time_context(request.message)
            
            if not self.enabled:
                return self._fallback_response(request.message, real_time_context)
            
            # 2. Build messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # 3. Add Real-time Data to Context
            if real_time_context:
                data_msg = f"REAL-TIME AWS DATA FOR CURRENT USER:\n{real_time_context}"
                messages.append({"role": "system", "content": data_msg})
            
            # Add general context (form data, etc.)
            if request.context:
                messages.append({"role": "system", "content": self._build_context_message(request.context)})
            
            # Add history
            if request.conversation_history:
                for msg in request.conversation_history[-5:]:
                    messages.append({"role": msg.role, "content": msg.content})
            
            # Add user message
            messages.append({"role": "user", "content": request.message})
            
            # Define Functions
            functions = [
                {
                    "name": "create_auto_scaling_rule",
                    "description": "Pre-authorize an automatic scaling rule that triggers when metrics cross a threshold.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the rule"},
                            "service_type": {"type": "string", "enum": ["ebs", "ec2", "rds"]},
                            "metric_name": {"type": "string", "description": "CloudWatch metric (e.g., VolumeQueueLength, CPUUtilization)"},
                            "threshold_value": {"type": "number", "description": "The metric value that triggers the rule"},
                            "action_amount_gb": {"type": "number", "description": "For EBS, amount of GB to add. Default is 4"},
                            "target_instance_type": {"type": "string", "description": "For EC2, target instance size"},
                            "target_db_instance_class": {"type": "string", "description": "For RDS, target DB class"}
                        },
                        "required": ["name", "service_type", "metric_name", "threshold_value"]
                    }
                },
                {
                    "name": "execute_scaling_action",
                    "description": "Immediately trigger a scaling action for a specific AWS resource.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_type": {"type": "string", "enum": ["ebs", "ec2", "rds"]},
                            "resource_id": {"type": "string", "description": "The AWS resource ID (e.g., vol-1234, i-5678)"},
                            "action_amount_gb": {"type": "number", "description": "For EBS, amount of GB to add"},
                            "target_instance_type": {"type": "string", "description": "For EC2, target instance size"},
                            "target_db_instance_class": {"type": "string", "description": "For RDS, target instance class"}
                        },
                        "required": ["service_type", "resource_id"]
                    }
                }
            ]
            
            # 4. Call OpenAI
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                functions=functions,
                function_call="auto",
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            response_message = response.choices[0].message
            
            # 5. Handle Function Calls
            if response_message.get("function_call"):
                function_name = response_message["function_call"]["name"]
                
                try:
                    function_args = json.loads(response_message["function_call"]["arguments"])
                    result_text = await self._handle_function_call(function_name, function_args)
                except Exception as e:
                    logger.error(f"Function call error: {e}")
                    result_text = "Sorry, I ran into an error trying to execute that action."
                
                # Let the AI know the result so it can summarize it
                messages.append(response_message)
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": result_text
                })
                
                second_response = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens
                )
                assistant_message = second_response.choices[0].message.content.strip()
            else:
                assistant_message = response_message.content.strip()
            suggestions = self._generate_suggestions(request.context)
            
            return ChatResponse(
                message=assistant_message,
                suggestions=suggestions,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Platform AI assistant error", error=str(e))
            return self._fallback_response(request.message, self._get_real_time_context(request.message))

    def _get_real_time_context(self, query: str) -> str:
        """Fetch pertinent data from AWSDataService based on the query."""
        if not self.aws_service:
            return "AWS is not currently connected."
            
        q = query.lower()
        context_parts = []
        
        try:
            # Instance counts
            if any(k in q for k in ["instance", "ec2", "running", "server"]):
                summary = self.aws_service.get_dashboard_summary()
                ec2_count = summary.get("active_instances", 0)
                context_parts.append(f"Active EC2 Instances: {ec2_count}")
                
            # Cost info
            if any(k in q for k in ["cost", "spend", "bill", "money"]):
                summary = self.aws_service.get_dashboard_summary()
                monthly = summary.get("monthly_spend", 0)
                savings = summary.get("potential_savings", 0)
                context_parts.append(f"Current Monthly Spend: ${monthly}")
                context_parts.append(f"Potential Savings Identified: ${savings}")
                
            # S3/Buckets
            if "bucket" in q or "s3" in q:
                # Fallback to general summary if specific S3 method isn't called
                context_parts.append("The user is asking about S3. Mention that you can see their buckets in the Compliance section.")

        except Exception as e:
            logger.warning("Failed to fetch real-time context", error=str(e))
            
        return "\n".join(context_parts) if context_parts else ""

    async def _handle_function_call(self, function_name: str, args: Dict) -> str:
        engine = get_engine()
        
        if function_name == "create_auto_scaling_rule":
            action = {}
            if args["service_type"] == "ebs":
                action = {"action": "increase_storage", "amount_gb": args.get("action_amount_gb", 4)}
            elif args["service_type"] == "ec2":
                action = {"action": "resize_instance", "target_instance_type": args.get("target_instance_type", "m5.xlarge")}
            elif args["service_type"] == "rds":
                action = {"action": "resize_db_instance", "target_db_instance_class": args.get("target_db_instance_class", "db.m5.large")}
                
            rule_data = {
                "name": args["name"],
                "service_type": args["service_type"],
                "metric_name": args["metric_name"],
                "metric_namespace": "AWS/EC2" if args["service_type"] == "ec2" else f"AWS/{args['service_type'].upper()}",
                "metric_dimension_name": "InstanceId" if args["service_type"] == "ec2" else ("VolumeId" if args["service_type"] == "ebs" else "DBInstanceIdentifier"),
                "threshold_value": args["threshold_value"],
                "scaling_action": action,
            }
            rule = engine.create_rule(rule_data)
            return json.dumps({"status": "success", "message": f"Created rule '{rule['name']}' with ID {rule['id']}"})
            
        elif function_name == "execute_scaling_action":
            # For immediate execution, we create a temporary rule format and trigger the engine directly
            service = args["service_type"]
            action = {}
            if service == "ebs":
                action = {"action": "increase_storage", "amount_gb": args.get("action_amount_gb", 4)}
            elif service == "ec2":
                action = {"action": "resize_instance", "target_instance_type": args.get("target_instance_type", "m5.xlarge")}
            elif service == "rds":
                action = {"action": "resize_db_instance", "target_db_instance_class": args.get("target_db_instance_class", "db.m5.large")}
                
            dummy_rule = {
                "id": "manual-chat-trigger",
                "name": f"Manual scaling requested by User for {args['resource_id']}",
                "service_type": service,
                "scaling_direction": "scale_up",
                "scaling_action": action,
            }
            res = engine._execute_scaling(dummy_rule, args["resource_id"], 0.0)
            return json.dumps({"status": res.get("status", "unknown"), "details": res})
            
        return json.dumps({"error": "Unknown function"})

    def _build_context_message(self, context: Dict) -> str:
        """Detailed context builder for the whole platform."""
        parts = []
        if "page" in context:
            parts.append(f"User is currently on the '{context['page']}' page.")
        if "migration_workload" in context:
            parts.append(f"Analyzing migration for: {context['migration_workload']}")
        return " | ".join(parts)

    def _generate_suggestions(self, context: Optional[Dict]) -> List[str]:
        """Context-aware suggestions for the platform."""
        page = context.get("page", "").lower() if context else ""
        
        if "cost" in page:
            return ["Show potential savings", "What are anomalies?", "Export cost report"]
        if "migration" in page:
            return ["How is TCO calculated?", "Risk assessment summary", "Next migration steps"]
        if "scheduler" in page:
            return ["Set a new schedule", "Why use RDS scheduling?", "Manual override"]
            
        return [
            "How many instances am I running?",
            "How can I save money on AWS?",
            "Tell me about migration planning."
        ]

    def _fallback_response(self, message: str, real_time_context: str) -> ChatResponse:
        """Enhanced fallback logic that still uses real-time data even without OpenAI."""
        q = message.lower()
        
        if real_time_context and any(k in q for k in ["how many", "count", "current"]):
            prefix = "I've checked your live environment: "
            response = f"{prefix}\n{real_time_context}\n\n(Note: AI brain is currently in offline mode, but I can still read your live data!)"
        elif "hello" in q or "hi" in q:
            response = "Hello! I'm CloudPilot, your platform assistant. I can help with cost analysis, migration planning, and checking your live AWS resources."
        else:
            response = "I'm currently running in limited mode, but I can still answer basic questions. To enable my full AI capabilities, please provide an OpenAI API key in the backend environment."
            
        return ChatResponse(
            message=response,
            suggestions=["Check my instances", "How to save costs?", "Migration help"],
            timestamp=datetime.utcnow()
        )

    def get_contextual_help(self, field_name: str, current_value: Optional[str] = None) -> str:
        # Keeping existing help mapping for backwards compatibility
        help_texts = {
            "company_size": "Select your organization size for rightsizing tips.",
            "monthly_budget": "Set your target spend to enable anomaly alerts.",
        }
        return help_texts.get(field_name, "I can help explain this feature if you ask!")


# Global instance (will be re-initialized in start_backend with aws_service)
platform_assistant = PlatformAIAssistant()

