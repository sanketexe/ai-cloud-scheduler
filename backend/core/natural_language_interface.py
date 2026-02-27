"""
Natural Language Interface for Conversational AI
Provides advanced NLP capabilities for cost analysis and optimization queries
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import structlog
from pydantic import BaseModel, Field

# NLP and ML imports
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline
)
from sentence_transformers import SentenceTransformer
import openai

# Internal imports
from .models import CostData
from .exceptions import NLPProcessingError

logger = structlog.get_logger(__name__)


class IntentType(str, Enum):
    """Types of user intents for cost analysis queries"""
    COST_QUERY = "cost_query"
    OPTIMIZATION_REQUEST = "optimization_request"
    ANOMALY_INVESTIGATION = "anomaly_investigation"
    COMPARISON_REQUEST = "comparison_request"
    FORECAST_REQUEST = "forecast_request"
    RECOMMENDATION_REQUEST = "recommendation_request"
    DRILL_DOWN_REQUEST = "drill_down_request"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Types of entities that can be extracted from queries"""
    TIME_PERIOD = "time_period"
    CLOUD_SERVICE = "cloud_service"
    COST_AMOUNT = "cost_amount"
    RESOURCE_TYPE = "resource_type"
    REGION = "region"
    ACCOUNT = "account"
    TAG = "tag"
    METRIC = "metric"


@dataclass
class Entity:
    """Extracted entity from user query"""
    text: str
    entity_type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    normalized_value: Optional[Any] = None


@dataclass
class Intent:
    """Classified user intent"""
    intent_type: IntentType
    confidence: float
    entities: List[Entity] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response to a natural language query"""
    answer: str
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    confidence_score: float
    processing_time_ms: int


class ConversationContext(BaseModel):
    """Maintains conversation state and context"""
    conversation_id: str
    user_id: str
    session_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_focus: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    last_query_time: Optional[datetime] = None
    context_embeddings: List[float] = Field(default_factory=list)


class QueryParser:
    """Handles intent classification and entity extraction"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models_loaded = False
        self._intent_cache = {}  # Cache for intent classification results
        self._entity_cache = {}  # Cache for entity extraction results
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained NLP models"""
        if self._models_loaded:
            return
            
        try:
            # Intent classification model (using a general classification model)
            self.intent_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-medium"
            )
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/DialoGPT-medium",
                num_labels=len(IntentType)
            ).to(self.device)
            
            # Named Entity Recognition model
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentence embeddings for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self._models_loaded = True
            logger.info("NLP models loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load NLP models", error=str(e))
            raise NLPProcessingError(f"Model loading failed: {str(e)}")
    
    async def parse_query(self, query: str, context: Optional[ConversationContext] = None) -> Intent:
        """Parse natural language query to extract intent and entities"""
        try:
            start_time = datetime.now()
            
            # Check cache for similar queries (simple hash-based caching)
            query_hash = hash(query.lower().strip())
            if query_hash in self._intent_cache:
                cached_result = self._intent_cache[query_hash]
                logger.info(
                    "Query parsed from cache",
                    query=query[:100],
                    intent=cached_result.intent_type,
                    confidence=cached_result.confidence,
                    processing_time_ms=1  # Cache hit is very fast
                )
                return cached_result
            
            # Classify intent
            intent_type, intent_confidence = await self._classify_intent(query, context)
            
            # Extract entities (with caching)
            entities = await self._extract_entities(query)
            
            # Extract parameters based on intent and entities
            parameters = self._extract_parameters(query, intent_type, entities)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = Intent(
                intent_type=intent_type,
                confidence=intent_confidence,
                entities=entities,
                parameters=parameters
            )
            
            # Cache the result (limit cache size to prevent memory issues)
            if len(self._intent_cache) < 1000:  # Limit cache size
                self._intent_cache[query_hash] = result
            
            logger.info(
                "Query parsed successfully",
                query=query[:100],
                intent=intent_type,
                confidence=intent_confidence,
                entities_count=len(entities),
                processing_time_ms=processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error("Query parsing failed", query=query, error=str(e))
            return Intent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                entities=[],
                parameters={}
            )
    
    async def _classify_intent(self, query: str, context: Optional[ConversationContext]) -> Tuple[IntentType, float]:
        """Classify user intent using rule-based and ML approaches"""
        
        # Rule-based classification for common patterns
        query_lower = query.lower()
        
        # Cost query patterns
        if any(word in query_lower for word in ['cost', 'spend', 'bill', 'expense', 'price']):
            if any(word in query_lower for word in ['how much', 'what did', 'total']):
                return IntentType.COST_QUERY, 0.9
        
        # Optimization patterns
        if any(word in query_lower for word in ['optimize', 'reduce', 'save', 'cheaper', 'efficient', 'minimize', 'lower', 'cut', 'decrease']):
            return IntentType.OPTIMIZATION_REQUEST, 0.85
        
        # Anomaly patterns
        if any(word in query_lower for word in ['spike', 'unusual', 'anomaly', 'unexpected', 'why', 'strange', 'weird', 'odd', 'abnormal']):
            return IntentType.ANOMALY_INVESTIGATION, 0.8
        
        # Comparison patterns
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'better']):
            return IntentType.COMPARISON_REQUEST, 0.8
        
        # Forecast patterns
        if any(word in query_lower for word in ['predict', 'forecast', 'future', 'next month', 'will']):
            return IntentType.FORECAST_REQUEST, 0.8
        
        # Recommendation patterns
        if any(word in query_lower for word in ['recommend', 'suggest', 'should', 'best']):
            return IntentType.RECOMMENDATION_REQUEST, 0.8
        
        # Drill-down patterns
        if any(word in query_lower for word in ['breakdown', 'details', 'drill down', 'analyze']):
            return IntentType.DRILL_DOWN_REQUEST, 0.8
        
        # Use context if available
        if context and context.current_focus:
            if 'cost' in context.current_focus:
                return IntentType.COST_QUERY, 0.6
            elif 'optimization' in context.current_focus:
                return IntentType.OPTIMIZATION_REQUEST, 0.6
        
        return IntentType.UNKNOWN, 0.3
    
    async def _extract_entities(self, query: str) -> List[Entity]:
        """Extract entities from the query using NER and custom patterns"""
        entities = []
        
        try:
            # Use NER pipeline for general entities
            ner_results = self.ner_pipeline(query)
            
            for result in ner_results:
                entity_type = self._map_ner_label_to_entity_type(result['entity_group'])
                if entity_type:
                    entities.append(Entity(
                        text=result['word'],
                        entity_type=entity_type,
                        confidence=result['score'],
                        start_pos=result['start'],
                        end_pos=result['end']
                    ))
            
            # Custom pattern matching for domain-specific entities
            entities.extend(self._extract_custom_entities(query))
            
        except Exception as e:
            logger.warning("Entity extraction failed", error=str(e))
        
        return entities
    
    def _map_ner_label_to_entity_type(self, ner_label: str) -> Optional[EntityType]:
        """Map NER labels to our entity types"""
        mapping = {
            'ORG': EntityType.CLOUD_SERVICE,
            'MISC': EntityType.RESOURCE_TYPE,
            'LOC': EntityType.REGION,
        }
        return mapping.get(ner_label)
    
    def _extract_custom_entities(self, query: str) -> List[Entity]:
        """Extract domain-specific entities using regex patterns"""
        entities = []
        
        # Time period patterns
        time_patterns = [
            (r'\b(last|past)\s+(\d+)\s+(day|week|month|year)s?\b', EntityType.TIME_PERIOD),
            (r'\b(this|current)\s+(week|month|year)\b', EntityType.TIME_PERIOD),
            (r'\b(yesterday|today)\b', EntityType.TIME_PERIOD),
            (r'\b\d{4}-\d{2}-\d{2}\b', EntityType.TIME_PERIOD),
        ]
        
        # Cost amount patterns
        cost_patterns = [
            (r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b', EntityType.COST_AMOUNT),
            (r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars?\b', EntityType.COST_AMOUNT),
        ]
        
        # Cloud service patterns
        service_patterns = [
            (r'\b(ec2|s3|rds|lambda|cloudfront|elb|ebs)\b', EntityType.CLOUD_SERVICE),
            (r'\b(compute engine|cloud storage|bigquery|cloud sql)\b', EntityType.CLOUD_SERVICE),
            (r'\b(virtual machines|blob storage|cosmos db|app service)\b', EntityType.CLOUD_SERVICE),
        ]
        
        # Region patterns
        region_patterns = [
            (r'\b(us-east-1|us-west-2|eu-west-1|ap-southeast-1)\b', EntityType.REGION),
            (r'\b(virginia|oregon|ireland|singapore)\b', EntityType.REGION),
        ]
        
        all_patterns = time_patterns + cost_patterns + service_patterns + region_patterns
        
        for pattern, entity_type in all_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(),
                    entity_type=entity_type,
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=self._normalize_entity_value(match.group(), entity_type)
                ))
        
        return entities
    
    def _normalize_entity_value(self, text: str, entity_type: EntityType) -> Any:
        """Normalize entity values for consistent processing"""
        if entity_type == EntityType.COST_AMOUNT:
            # Extract numeric value from cost strings
            numeric_match = re.search(r'[\d,]+(?:\.\d{2})?', text.replace('$', ''))
            if numeric_match:
                return float(numeric_match.group().replace(',', ''))
        
        elif entity_type == EntityType.TIME_PERIOD:
            # Normalize time periods to standard format
            if 'last' in text.lower() or 'past' in text.lower():
                return text.lower()
            elif 'this' in text.lower() or 'current' in text.lower():
                return text.lower()
        
        return text.lower()
    
    def clear_caches(self):
        """Clear intent and entity caches"""
        self._intent_cache.clear()
        self._entity_cache.clear()
        logger.info("NLP caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "intent_cache_size": len(self._intent_cache),
            "entity_cache_size": len(self._entity_cache)
        }
    
    def _extract_parameters(self, query: str, intent_type: IntentType, entities: List[Entity]) -> Dict[str, Any]:
        """Extract parameters based on intent and entities"""
        parameters = {}
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            if entity.entity_type not in entity_groups:
                entity_groups[entity.entity_type] = []
            entity_groups[entity.entity_type].append(entity)
        
        # Extract parameters based on intent
        if intent_type == IntentType.COST_QUERY:
            parameters['query_type'] = 'cost_analysis'
            if EntityType.TIME_PERIOD in entity_groups:
                parameters['time_period'] = entity_groups[EntityType.TIME_PERIOD][0].normalized_value
            if EntityType.CLOUD_SERVICE in entity_groups:
                parameters['services'] = [e.normalized_value for e in entity_groups[EntityType.CLOUD_SERVICE]]
        
        elif intent_type == IntentType.OPTIMIZATION_REQUEST:
            parameters['query_type'] = 'optimization'
            parameters['optimization_focus'] = self._extract_optimization_focus(query)
        
        elif intent_type == IntentType.ANOMALY_INVESTIGATION:
            parameters['query_type'] = 'anomaly_analysis'
            if EntityType.TIME_PERIOD in entity_groups:
                parameters['time_period'] = entity_groups[EntityType.TIME_PERIOD][0].normalized_value
        
        return parameters
    
    def _extract_optimization_focus(self, query: str) -> str:
        """Extract the focus area for optimization requests"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compute', 'instance', 'server']):
            return 'compute'
        elif any(word in query_lower for word in ['storage', 'disk', 'volume']):
            return 'storage'
        elif any(word in query_lower for word in ['network', 'bandwidth', 'data transfer']):
            return 'network'
        elif any(word in query_lower for word in ['database', 'db', 'rds']):
            return 'database'
        else:
            return 'general'


class ResponseGenerator:
    """Generates context-aware responses with visualizations"""
    
    def __init__(self):
        self._openai_client = None
        
    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client"""
        if self._openai_client is None:
            try:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self._openai_client = openai.AsyncOpenAI(api_key=api_key)
                else:
                    logger.warning("OPENAI_API_KEY not set, using fallback responses")
                    self._openai_client = None
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self._openai_client = None
        return self._openai_client
        
    async def generate_response(
        self,
        intent: Intent,
        data: Optional[Dict[str, Any]] = None,
        context: Optional[ConversationContext] = None
    ) -> QueryResponse:
        """Generate a comprehensive response to the user query"""
        
        start_time = datetime.now()
        
        try:
            # Generate text response
            answer = await self._generate_text_response(intent, data, context)
            
            # Generate visualizations
            visualizations = self._generate_visualizations(intent, data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(intent, data)
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(intent, context)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return QueryResponse(
                answer=answer,
                visualizations=visualizations,
                recommendations=recommendations,
                follow_up_questions=follow_up_questions,
                confidence_score=intent.confidence,
                processing_time_ms=int(processing_time)
            )
            
        except Exception as e:
            logger.error("Response generation failed", error=str(e))
            return QueryResponse(
                answer="I apologize, but I encountered an error processing your request. Please try rephrasing your question.",
                confidence_score=0.0,
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    async def _generate_text_response(
        self,
        intent: Intent,
        data: Optional[Dict[str, Any]],
        context: Optional[ConversationContext]
    ) -> str:
        """Generate the main text response using AI"""
        
        # Build context for the AI model
        system_prompt = self._build_system_prompt(intent.intent_type)
        user_prompt = self._build_user_prompt(intent, data, context)
        
        try:
            if self.openai_client is None:
                return self._generate_fallback_response(intent, data)
                
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning("OpenAI API failed, using fallback response", error=str(e))
            return self._generate_fallback_response(intent, data)
    
    def _build_system_prompt(self, intent_type: IntentType) -> str:
        """Build system prompt based on intent type"""
        
        base_prompt = """You are an expert FinOps analyst and cloud cost optimization specialist. 
        You help users understand their cloud spending and provide actionable insights for cost optimization.
        
        Guidelines:
        - Be concise but informative (2-3 sentences)
        - Use specific numbers and data when available
        - Provide actionable recommendations
        - Explain technical concepts in simple terms
        - Focus on cost impact and business value
        """
        
        intent_specific = {
            IntentType.COST_QUERY: "Focus on providing clear cost breakdowns and explaining spending patterns.",
            IntentType.OPTIMIZATION_REQUEST: "Prioritize specific, actionable cost optimization recommendations.",
            IntentType.ANOMALY_INVESTIGATION: "Explain potential root causes and provide investigation steps.",
            IntentType.COMPARISON_REQUEST: "Provide clear comparisons with pros/cons and cost implications.",
            IntentType.FORECAST_REQUEST: "Explain forecasting methodology and confidence levels.",
            IntentType.RECOMMENDATION_REQUEST: "Provide prioritized recommendations with expected impact.",
            IntentType.DRILL_DOWN_REQUEST: "Provide detailed analysis with supporting data."
        }
        
        return base_prompt + "\n\n" + intent_specific.get(intent_type, "")
    
    def _build_user_prompt(
        self,
        intent: Intent,
        data: Optional[Dict[str, Any]],
        context: Optional[ConversationContext]
    ) -> str:
        """Build user prompt with context and data"""
        
        prompt_parts = []
        
        # Add intent information
        prompt_parts.append(f"User intent: {intent.intent_type.value}")
        prompt_parts.append(f"Confidence: {intent.confidence:.2f}")
        
        # Add extracted entities
        if intent.entities:
            entities_text = ", ".join([f"{e.entity_type.value}: {e.text}" for e in intent.entities])
            prompt_parts.append(f"Extracted entities: {entities_text}")
        
        # Add parameters
        if intent.parameters:
            params_text = ", ".join([f"{k}: {v}" for k, v in intent.parameters.items()])
            prompt_parts.append(f"Parameters: {params_text}")
        
        # Add data context
        if data:
            prompt_parts.append(f"Available data: {json.dumps(data, default=str)[:500]}...")
        
        # Add conversation context
        if context and context.current_focus:
            prompt_parts.append(f"Current conversation focus: {context.current_focus}")
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_response(self, intent: Intent, data: Optional[Dict[str, Any]]) -> str:
        """Generate fallback response when AI is unavailable"""
        
        fallback_responses = {
            IntentType.COST_QUERY: "I can help you analyze your cloud costs. Please provide more specific details about the time period or services you'd like to examine.",
            IntentType.OPTIMIZATION_REQUEST: "I can suggest cost optimization strategies. Common areas include rightsizing instances, using reserved instances, and optimizing storage.",
            IntentType.ANOMALY_INVESTIGATION: "Cost anomalies can be caused by unexpected usage spikes, new resources, or pricing changes. Let me help you investigate the specific timeframe.",
            IntentType.COMPARISON_REQUEST: "I can help compare different options. Please specify what you'd like to compare (services, time periods, or configurations).",
            IntentType.FORECAST_REQUEST: "Cost forecasting considers historical trends and planned changes. I can help predict future spending based on your usage patterns.",
            IntentType.RECOMMENDATION_REQUEST: "I can provide cost optimization recommendations based on your current usage. Common strategies include reserved instances and rightsizing.",
            IntentType.DRILL_DOWN_REQUEST: "I can provide detailed cost breakdowns by service, region, or time period. What specific area would you like to analyze?"
        }
        
        return fallback_responses.get(intent.intent_type, "I'm here to help with your cloud cost analysis. Please let me know what specific information you need.")
    
    def _generate_visualizations(self, intent: Intent, data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate visualization specifications based on intent and data"""
        
        visualizations = []
        
        if intent.intent_type == IntentType.COST_QUERY:
            visualizations.extend([
                {
                    "type": "line_chart",
                    "title": "Cost Trend Over Time",
                    "data_source": "cost_timeseries",
                    "x_axis": "date",
                    "y_axis": "cost",
                    "description": "Daily cost trends for the requested period"
                },
                {
                    "type": "pie_chart",
                    "title": "Cost Breakdown by Service",
                    "data_source": "service_costs",
                    "description": "Distribution of costs across different cloud services"
                }
            ])
        
        elif intent.intent_type == IntentType.OPTIMIZATION_REQUEST:
            visualizations.extend([
                {
                    "type": "bar_chart",
                    "title": "Optimization Opportunities",
                    "data_source": "optimization_recommendations",
                    "x_axis": "recommendation",
                    "y_axis": "potential_savings",
                    "description": "Potential cost savings from optimization recommendations"
                }
            ])
        
        elif intent.intent_type == IntentType.ANOMALY_INVESTIGATION:
            visualizations.extend([
                {
                    "type": "anomaly_chart",
                    "title": "Cost Anomalies Detection",
                    "data_source": "anomaly_data",
                    "description": "Identified cost anomalies with confidence scores"
                }
            ])
        
        return visualizations
    
    def _generate_recommendations(self, intent: Intent, data: Optional[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on intent"""
        
        recommendations = []
        
        if intent.intent_type == IntentType.COST_QUERY:
            recommendations.extend([
                "Set up cost alerts to monitor spending thresholds",
                "Review resource utilization to identify optimization opportunities",
                "Consider reserved instances for predictable workloads"
            ])
        
        elif intent.intent_type == IntentType.OPTIMIZATION_REQUEST:
            focus = intent.parameters.get('optimization_focus', 'general')
            
            if focus == 'compute':
                recommendations.extend([
                    "Rightsize underutilized instances",
                    "Use spot instances for fault-tolerant workloads",
                    "Implement auto-scaling to match demand"
                ])
            elif focus == 'storage':
                recommendations.extend([
                    "Move infrequently accessed data to cheaper storage tiers",
                    "Enable compression and deduplication",
                    "Review and delete unused snapshots"
                ])
            else:
                recommendations.extend([
                    "Analyze resource utilization patterns",
                    "Implement cost allocation tags",
                    "Review and optimize data transfer costs"
                ])
        
        elif intent.intent_type == IntentType.ANOMALY_INVESTIGATION:
            recommendations.extend([
                "Check for new resource deployments during the anomaly period",
                "Review usage patterns for unusual spikes",
                "Verify pricing changes or billing adjustments"
            ])
        
        # Fallback recommendations for unknown intents
        elif intent.intent_type == IntentType.UNKNOWN:
            recommendations.extend([
                "Review your current cloud spending patterns",
                "Set up cost monitoring and alerts",
                "Consider implementing cost optimization strategies"
            ])
        
        return recommendations
    
    def _generate_follow_up_questions(self, intent: Intent, context: Optional[ConversationContext]) -> List[str]:
        """Generate relevant follow-up questions"""
        
        questions = []
        
        if intent.intent_type == IntentType.COST_QUERY:
            questions.extend([
                "Would you like to see a breakdown by specific services?",
                "Should I analyze cost trends for a different time period?",
                "Are you interested in optimization recommendations for these costs?"
            ])
        
        elif intent.intent_type == IntentType.OPTIMIZATION_REQUEST:
            questions.extend([
                "Would you like me to prioritize recommendations by potential savings?",
                "Should I focus on specific services or resource types?",
                "Are you interested in the implementation timeline for these optimizations?"
            ])
        
        elif intent.intent_type == IntentType.ANOMALY_INVESTIGATION:
            questions.extend([
                "Would you like me to investigate a specific service or resource?",
                "Should I check for similar anomalies in other time periods?",
                "Are you interested in setting up alerts to prevent future anomalies?"
            ])
        
        # Fallback questions for unknown intents
        elif intent.intent_type == IntentType.UNKNOWN:
            questions.extend([
                "Would you like to analyze your current cloud costs?",
                "Are you interested in cost optimization recommendations?",
                "Should I help you investigate any cost anomalies?"
            ])
        
        return questions[:3]  # Limit to 3 questions


class ContextManager:
    """Manages conversation state and context"""
    
    def __init__(self):
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def get_context(self, conversation_id: str, user_id: str) -> ConversationContext:
        """Get or create conversation context"""
        
        if conversation_id not in self.active_contexts:
            self.active_contexts[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id
            )
        
        return self.active_contexts[conversation_id]
    
    async def update_context(
        self,
        conversation_id: str,
        query: str,
        intent: Intent,
        response: QueryResponse
    ) -> ConversationContext:
        """Update conversation context with new interaction"""
        
        context = self.active_contexts.get(conversation_id)
        if not context:
            return await self.get_context(conversation_id, "unknown")
        
        # Add interaction to history
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "intent": intent.intent_type.value,
            "response_summary": response.answer[:100] + "..." if len(response.answer) > 100 else response.answer,
            "confidence": intent.confidence
        }
        
        context.session_history.append(interaction)
        
        # Update current focus based on intent
        context.current_focus = self._determine_focus(intent, context)
        
        # Update context embeddings for semantic similarity
        context.context_embeddings = self._update_embeddings(query, context)
        
        context.last_query_time = datetime.now()
        
        # Limit history size
        if len(context.session_history) > 20:
            context.session_history = context.session_history[-20:]
        
        return context
    
    def _determine_focus(self, intent: Intent, context: ConversationContext) -> str:
        """Determine the current conversation focus"""
        
        # Use intent type as primary focus
        focus_mapping = {
            IntentType.COST_QUERY: "cost_analysis",
            IntentType.OPTIMIZATION_REQUEST: "optimization",
            IntentType.ANOMALY_INVESTIGATION: "anomaly_investigation",
            IntentType.COMPARISON_REQUEST: "comparison",
            IntentType.FORECAST_REQUEST: "forecasting",
            IntentType.RECOMMENDATION_REQUEST: "recommendations",
            IntentType.DRILL_DOWN_REQUEST: "detailed_analysis"
        }
        
        new_focus = focus_mapping.get(intent.intent_type, "general")
        
        # Consider conversation history for focus continuity
        if context.session_history:
            recent_intents = [interaction.get("intent") for interaction in context.session_history[-3:]]
            if len(set(recent_intents)) == 1:  # All recent intents are the same
                return new_focus
        
        return new_focus
    
    def _update_embeddings(self, query: str, context: ConversationContext) -> List[float]:
        """Update context embeddings for semantic similarity"""
        
        try:
            # Combine recent queries for context
            recent_queries = [query]
            if context.session_history:
                recent_queries.extend([
                    interaction.get("query", "") 
                    for interaction in context.session_history[-3:]
                ])
            
            # Generate embeddings for the combined context
            combined_text = " ".join(recent_queries)
            embeddings = self.sentence_model.encode(combined_text)
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.warning("Failed to update context embeddings", error=str(e))
            return []
    
    async def cleanup_old_contexts(self, max_age_hours: int = 24):
        """Clean up old conversation contexts"""
        
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        contexts_to_remove = []
        for conversation_id, context in self.active_contexts.items():
            if context.last_query_time and context.last_query_time.timestamp() < cutoff_time:
                contexts_to_remove.append(conversation_id)
        
        for conversation_id in contexts_to_remove:
            del self.active_contexts[conversation_id]
        
        logger.info(f"Cleaned up {len(contexts_to_remove)} old conversation contexts")


class NaturalLanguageInterface:
    """Main interface for natural language processing"""
    
    def __init__(self):
        self.query_parser = QueryParser()
        self.response_generator = ResponseGenerator()
        self.context_manager = ContextManager()
        
    async def process_query(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        data_context: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """Process a natural language query and return a comprehensive response"""
        
        start_time = datetime.now()
        
        try:
            # Get conversation context
            context = await self.context_manager.get_context(conversation_id, user_id)
            
            # Parse the query
            intent = await self.query_parser.parse_query(query, context)
            
            # Generate response
            response = await self.response_generator.generate_response(
                intent, data_context, context
            )
            
            # Update context
            await self.context_manager.update_context(
                conversation_id, query, intent, response
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(
                "Query processed successfully",
                conversation_id=conversation_id,
                user_id=user_id,
                intent=intent.intent_type.value,
                confidence=intent.confidence,
                processing_time_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Query processing failed",
                conversation_id=conversation_id,
                query=query[:100],
                error=str(e)
            )
            
            # Return error response
            return QueryResponse(
                answer="I apologize, but I encountered an error processing your request. Please try again or rephrase your question.",
                confidence_score=0.0,
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    async def generate_insights(
        self,
        data: Dict[str, Any],
        query_intent: Intent
    ) -> QueryResponse:
        """Generate insights from data based on query intent"""
        
        return await self.response_generator.generate_response(
            query_intent, data, None
        )
    
    async def create_visualization(
        self,
        data: Any,
        viz_type: str
    ) -> Dict[str, Any]:
        """Create visualization specification for given data"""
        
        visualization_specs = {
            "line_chart": {
                "type": "line",
                "data": data,
                "options": {
                    "responsive": True,
                    "scales": {
                        "x": {"type": "time"},
                        "y": {"beginAtZero": True}
                    }
                }
            },
            "bar_chart": {
                "type": "bar",
                "data": data,
                "options": {
                    "responsive": True,
                    "scales": {
                        "y": {"beginAtZero": True}
                    }
                }
            },
            "pie_chart": {
                "type": "pie",
                "data": data,
                "options": {
                    "responsive": True,
                    "plugins": {
                        "legend": {"position": "right"}
                    }
                }
            }
        }
        
        return visualization_specs.get(viz_type, {"type": viz_type, "data": data})
    
    async def maintain_context(
        self,
        conversation_id: str,
        interaction: Dict[str, Any]
    ) -> ConversationContext:
        """Maintain conversation context with new interaction"""
        
        context = self.context_manager.active_contexts.get(conversation_id)
        if context:
            context.session_history.append(interaction)
            context.last_query_time = datetime.now()
        
        return context
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the NLP interface"""
        cache_stats = self.query_parser.get_cache_stats()
        context_stats = {
            "active_conversations": len(self.context_manager.active_contexts),
            "total_contexts": len(self.context_manager.active_contexts)
        }
        
        return {
            "cache_stats": cache_stats,
            "context_stats": context_stats,
            "models_loaded": hasattr(self.query_parser, '_models_loaded') and self.query_parser._models_loaded
        }
    
    async def clear_caches(self):
        """Clear all caches to free memory"""
        self.query_parser.clear_caches()
        await self.context_manager.cleanup_old_contexts(max_age_hours=1)  # Aggressive cleanup
        logger.info("All NLP caches cleared")


# Global instance - lazy initialization
_natural_language_interface = None

def get_natural_language_interface() -> NaturalLanguageInterface:
    """Get the global natural language interface instance"""
    global _natural_language_interface
    if _natural_language_interface is None:
        _natural_language_interface = NaturalLanguageInterface()
    return _natural_language_interface

# For backward compatibility
natural_language_interface = get_natural_language_interface()