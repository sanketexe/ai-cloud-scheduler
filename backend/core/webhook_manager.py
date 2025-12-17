"""
Webhook Manager for External System Integration

Provides real-time event streaming and webhook endpoint management
for automated cost optimization actions and monitoring.
"""

import json
import logging
import asyncio
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import uuid

import requests
import structlog
from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from .database import Base, get_db_session
from .automation_models import OptimizationAction, ActionStatus
from .exceptions import WebhookException, ValidationException

logger = structlog.get_logger(__name__)


class WebhookEventType(Enum):
    """Types of webhook events"""
    ACTION_CREATED = "action.created"
    ACTION_SCHEDULED = "action.scheduled"
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"
    ACTION_ROLLED_BACK = "action.rolled_back"
    SAFETY_CHECK_FAILED = "safety_check.failed"
    POLICY_VIOLATION = "policy.violation"
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"
    SYSTEM_ERROR = "system.error"
    SAVINGS_CALCULATED = "savings.calculated"


class WebhookStatus(Enum):
    """Webhook endpoint status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    SUSPENDED = "suspended"


class WebhookSecurityType(Enum):
    """Webhook security methods"""
    NONE = "none"
    BEARER_TOKEN = "bearer_token"
    HMAC_SHA256 = "hmac_sha256"
    BASIC_AUTH = "basic_auth"


@dataclass
class WebhookEvent:
    """Webhook event data structure"""
    event_type: WebhookEventType
    event_id: str
    timestamp: datetime
    data: Dict[str, Any]
    resource_id: Optional[str] = None
    action_id: Optional[str] = None
    correlation_id: Optional[str] = None
    retry_count: int = 0


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""
    id: str
    name: str
    url: str
    event_types: List[WebhookEventType]
    security_type: WebhookSecurityType
    security_config: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5
    is_active: bool = True


class WebhookEndpointModel(Base):
    """SQLAlchemy model for webhook endpoints"""
    __tablename__ = "webhook_endpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    url = Column(String(500), nullable=False)
    event_types = Column(JSONB, nullable=False, default=list)
    security_type = Column(String(50), nullable=False, default="none")
    security_config = Column(JSONB, nullable=False, default=dict)
    headers = Column(JSONB, nullable=False, default=dict)
    timeout = Column(Integer, nullable=False, default=30)
    retry_attempts = Column(Integer, nullable=False, default=3)
    retry_delay = Column(Integer, nullable=False, default=5)
    is_active = Column(Boolean, nullable=False, default=True)
    status = Column(String(50), nullable=False, default="active")
    last_success = Column(DateTime(timezone=True))
    last_failure = Column(DateTime(timezone=True))
    failure_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(255), nullable=False)


class WebhookDelivery(Base):
    """SQLAlchemy model for webhook delivery attempts"""
    __tablename__ = "webhook_deliveries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    endpoint_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    event_id = Column(String(255), nullable=False, index=True)
    event_type = Column(String(100), nullable=False)
    payload = Column(JSONB, nullable=False)
    response_status = Column(Integer)
    response_body = Column(Text)
    response_headers = Column(JSONB)
    delivery_time = Column(DateTime(timezone=True))
    attempt_number = Column(Integer, nullable=False, default=1)
    success = Column(Boolean, nullable=False, default=False)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class WebhookManager:
    """
    Manages webhook endpoints and event delivery for external system integration.
    Provides real-time event streaming with proper formatting and security.
    """
    
    def __init__(self):
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.event_handlers: Dict[WebhookEventType, List[Callable]] = {}
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self._delivery_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the webhook delivery service"""
        if self.is_running:
            return
        
        self.is_running = True
        self._delivery_task = asyncio.create_task(self._delivery_worker())
        logger.info("Webhook manager started")
    
    async def stop(self):
        """Stop the webhook delivery service"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._delivery_task:
            self._delivery_task.cancel()
            try:
                await self._delivery_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Webhook manager stopped")
    
    def register_endpoint(self, endpoint: WebhookEndpoint) -> str:
        """Register a new webhook endpoint"""
        # Validate URL
        parsed_url = urlparse(endpoint.url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValidationException(f"Invalid webhook URL: {endpoint.url}")
        
        # Validate security configuration
        self._validate_security_config(endpoint.security_type, endpoint.security_config)
        
        # Store endpoint
        self.endpoints[endpoint.id] = endpoint
        
        # Persist to database
        with next(get_db_session()) as db:
            endpoint_uuid = uuid.UUID(endpoint.id) if isinstance(endpoint.id, str) else endpoint.id
            db_endpoint = WebhookEndpointModel(
                id=endpoint_uuid,
                name=endpoint.name,
                url=endpoint.url,
                event_types=[event.value for event in endpoint.event_types],
                security_type=endpoint.security_type.value,
                security_config=endpoint.security_config,
                headers=endpoint.headers,
                timeout=endpoint.timeout,
                retry_attempts=endpoint.retry_attempts,
                retry_delay=endpoint.retry_delay,
                is_active=endpoint.is_active,
                created_by="system"
            )
            db.add(db_endpoint)
            db.commit()
        
        logger.info("Webhook endpoint registered", 
                   endpoint_id=endpoint.id, 
                   url=endpoint.url,
                   event_types=[e.value for e in endpoint.event_types])
        
        return endpoint.id
    
    def unregister_endpoint(self, endpoint_id: str):
        """Unregister a webhook endpoint"""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
        
        # Remove from database
        with next(get_db_session()) as db:
            endpoint_uuid = uuid.UUID(endpoint_id) if isinstance(endpoint_id, str) else endpoint_id
            db_endpoint = db.query(WebhookEndpointModel).filter(
                WebhookEndpointModel.id == endpoint_uuid
            ).first()
            if db_endpoint:
                db.delete(db_endpoint)
                db.commit()
        
        logger.info("Webhook endpoint unregistered", endpoint_id=endpoint_id)
    
    def update_endpoint_status(self, endpoint_id: str, status: WebhookStatus):
        """Update webhook endpoint status"""
        if endpoint_id in self.endpoints:
            self.endpoints[endpoint_id].is_active = (status == WebhookStatus.ACTIVE)
        
        # Update database
        with next(get_db_session()) as db:
            endpoint_uuid = uuid.UUID(endpoint_id) if isinstance(endpoint_id, str) else endpoint_id
            db.query(WebhookEndpointModel).filter(
                WebhookEndpointModel.id == endpoint_uuid
            ).update({
                "status": status.value,
                "is_active": status == WebhookStatus.ACTIVE
            })
            db.commit()
    
    async def emit_event(self, event: WebhookEvent):
        """Emit a webhook event to all registered endpoints"""
        if not self.is_running:
            logger.warning("Webhook manager not running, event dropped", 
                          event_type=event.event_type.value)
            return
        
        # Find endpoints that should receive this event
        target_endpoints = []
        for endpoint in self.endpoints.values():
            if (endpoint.is_active and 
                event.event_type in endpoint.event_types):
                target_endpoints.append(endpoint)
        
        if not target_endpoints:
            logger.debug("No endpoints registered for event type", 
                        event_type=event.event_type.value)
            return
        
        # Queue delivery for each endpoint
        for endpoint in target_endpoints:
            await self.delivery_queue.put((endpoint, event))
        
        logger.debug("Event queued for delivery", 
                    event_type=event.event_type.value,
                    endpoint_count=len(target_endpoints))
    
    async def _delivery_worker(self):
        """Background worker for webhook delivery"""
        while self.is_running:
            try:
                # Get next delivery from queue
                endpoint, event = await asyncio.wait_for(
                    self.delivery_queue.get(), timeout=1.0
                )
                
                # Attempt delivery
                await self._deliver_webhook(endpoint, event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error in webhook delivery worker", error=str(e))
    
    async def _deliver_webhook(self, endpoint: WebhookEndpoint, event: WebhookEvent):
        """Deliver webhook to a specific endpoint"""
        delivery_id = str(uuid.uuid4())
        
        # Prepare payload
        payload = self._format_webhook_payload(event)
        
        # Prepare headers
        headers = endpoint.headers.copy()
        headers.update({
            "Content-Type": "application/json",
            "User-Agent": "FinOps-Webhook/1.0",
            "X-Webhook-Event": event.event_type.value,
            "X-Webhook-Delivery": delivery_id,
            "X-Webhook-Timestamp": str(int(event.timestamp.timestamp()))
        })
        
        # Add security headers
        self._add_security_headers(headers, payload, endpoint.security_type, endpoint.security_config)
        
        # Attempt delivery with retries
        for attempt in range(1, endpoint.retry_attempts + 1):
            try:
                start_time = time.time()
                
                async with asyncio.timeout(endpoint.timeout):
                    response = await asyncio.to_thread(
                        requests.post,
                        endpoint.url,
                        json=payload,
                        headers=headers,
                        timeout=endpoint.timeout
                    )
                
                delivery_time = time.time() - start_time
                success = 200 <= response.status_code < 300
                
                # Log delivery attempt
                await self._log_delivery(
                    endpoint.id, event.event_id, event.event_type.value,
                    payload, response.status_code, response.text,
                    dict(response.headers), delivery_time, attempt, success
                )
                
                if success:
                    await self._update_endpoint_success(endpoint.id)
                    logger.debug("Webhook delivered successfully",
                               endpoint_id=endpoint.id,
                               event_type=event.event_type.value,
                               status_code=response.status_code,
                               delivery_time=delivery_time)
                    return
                else:
                    logger.warning("Webhook delivery failed",
                                 endpoint_id=endpoint.id,
                                 event_type=event.event_type.value,
                                 status_code=response.status_code,
                                 attempt=attempt)
            
            except Exception as e:
                error_msg = str(e)
                
                # Log failed delivery attempt
                await self._log_delivery(
                    endpoint.id, event.event_id, event.event_type.value,
                    payload, None, None, None, None, attempt, False, error_msg
                )
                
                logger.error("Webhook delivery error",
                           endpoint_id=endpoint.id,
                           event_type=event.event_type.value,
                           attempt=attempt,
                           error=error_msg)
            
            # Wait before retry (except on last attempt)
            if attempt < endpoint.retry_attempts:
                await asyncio.sleep(endpoint.retry_delay)
        
        # All attempts failed
        await self._update_endpoint_failure(endpoint.id)
        logger.error("Webhook delivery failed after all retries",
                   endpoint_id=endpoint.id,
                   event_type=event.event_type.value)
    
    def _format_webhook_payload(self, event: WebhookEvent) -> Dict[str, Any]:
        """Format webhook payload with proper structure"""
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
            "resource_id": event.resource_id,
            "action_id": event.action_id,
            "correlation_id": event.correlation_id,
            "retry_count": event.retry_count,
            "version": "1.0"
        }
    
    def _add_security_headers(self, headers: Dict[str, str], payload: Dict[str, Any], 
                            security_type: WebhookSecurityType, security_config: Dict[str, Any]):
        """Add security headers based on configuration"""
        if security_type == WebhookSecurityType.BEARER_TOKEN:
            token = security_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif security_type == WebhookSecurityType.HMAC_SHA256:
            secret = security_config.get("secret")
            if secret:
                payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
                signature = hmac.new(
                    secret.encode('utf-8'),
                    payload_str.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        elif security_type == WebhookSecurityType.BASIC_AUTH:
            username = security_config.get("username")
            password = security_config.get("password")
            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
    
    def _validate_security_config(self, security_type: WebhookSecurityType, config: Dict[str, Any]):
        """Validate security configuration"""
        if security_type == WebhookSecurityType.BEARER_TOKEN:
            if not config.get("token"):
                raise ValidationException("Bearer token required for bearer_token security")
        
        elif security_type == WebhookSecurityType.HMAC_SHA256:
            if not config.get("secret"):
                raise ValidationException("Secret required for hmac_sha256 security")
        
        elif security_type == WebhookSecurityType.BASIC_AUTH:
            if not config.get("username") or not config.get("password"):
                raise ValidationException("Username and password required for basic_auth security")
    
    async def _log_delivery(self, endpoint_id: str, event_id: str, event_type: str,
                          payload: Dict[str, Any], status_code: Optional[int],
                          response_body: Optional[str], response_headers: Optional[Dict[str, str]],
                          delivery_time: Optional[float], attempt: int, success: bool,
                          error_message: Optional[str] = None):
        """Log webhook delivery attempt"""
        def _log():
            with next(get_db_session()) as db:
                endpoint_uuid = uuid.UUID(endpoint_id) if isinstance(endpoint_id, str) else endpoint_id
                delivery = WebhookDelivery(
                    endpoint_id=endpoint_uuid,
                    event_id=event_id,
                    event_type=event_type,
                    payload=payload,
                    response_status=status_code,
                    response_body=response_body,
                    response_headers=response_headers,
                    delivery_time=datetime.now() if delivery_time else None,
                    attempt_number=attempt,
                    success=success,
                    error_message=error_message
                )
                db.add(delivery)
                db.commit()
        
        await asyncio.to_thread(_log)
    
    async def _update_endpoint_success(self, endpoint_id: str):
        """Update endpoint after successful delivery"""
        def _update():
            with next(get_db_session()) as db:
                endpoint_uuid = uuid.UUID(endpoint_id) if isinstance(endpoint_id, str) else endpoint_id
                db.query(WebhookEndpointModel).filter(
                    WebhookEndpointModel.id == endpoint_uuid
                ).update({
                    "last_success": datetime.now(),
                    "failure_count": 0,
                    "status": WebhookStatus.ACTIVE.value
                })
                db.commit()
        
        await asyncio.to_thread(_update)
    
    async def _update_endpoint_failure(self, endpoint_id: str):
        """Update endpoint after failed delivery"""
        def _update():
            with next(get_db_session()) as db:
                endpoint_uuid = uuid.UUID(endpoint_id) if isinstance(endpoint_id, str) else endpoint_id
                endpoint = db.query(WebhookEndpointModel).filter(
                    WebhookEndpointModel.id == endpoint_uuid
                ).first()
                
                if endpoint:
                    endpoint.last_failure = datetime.now()
                    endpoint.failure_count += 1
                    
                    # Suspend endpoint after too many failures
                    if endpoint.failure_count >= 10:
                        endpoint.status = WebhookStatus.SUSPENDED.value
                        endpoint.is_active = False
                    
                    db.commit()
        
        await asyncio.to_thread(_update)
    
    def get_endpoint_status(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get webhook endpoint status and statistics"""
        with next(get_db_session()) as db:
            endpoint_uuid = uuid.UUID(endpoint_id) if isinstance(endpoint_id, str) else endpoint_id
            endpoint = db.query(WebhookEndpointModel).filter(
                WebhookEndpointModel.id == endpoint_uuid
            ).first()
            
            if not endpoint:
                return None
            
            # Get delivery statistics
            total_deliveries = db.query(WebhookDelivery).filter(
                WebhookDelivery.endpoint_id == endpoint.id
            ).count()
            
            successful_deliveries = db.query(WebhookDelivery).filter(
                WebhookDelivery.endpoint_id == endpoint.id,
                WebhookDelivery.success == True
            ).count()
            
            recent_deliveries = db.query(WebhookDelivery).filter(
                WebhookDelivery.endpoint_id == endpoint.id,
                WebhookDelivery.created_at >= datetime.now() - timedelta(hours=24)
            ).count()
            
            return {
                "endpoint_id": str(endpoint.id),
                "name": endpoint.name,
                "url": endpoint.url,
                "status": endpoint.status,
                "is_active": endpoint.is_active,
                "last_success": endpoint.last_success.isoformat() if endpoint.last_success else None,
                "last_failure": endpoint.last_failure.isoformat() if endpoint.last_failure else None,
                "failure_count": endpoint.failure_count,
                "total_deliveries": total_deliveries,
                "successful_deliveries": successful_deliveries,
                "success_rate": successful_deliveries / total_deliveries if total_deliveries > 0 else 0,
                "recent_deliveries_24h": recent_deliveries,
                "created_at": endpoint.created_at.isoformat()
            }
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all webhook endpoints with their status"""
        endpoints = []
        for endpoint_id in self.endpoints.keys():
            status = self.get_endpoint_status(endpoint_id)
            if status:
                endpoints.append(status)
        return endpoints


# Global webhook manager instance
_webhook_manager = None

def get_webhook_manager() -> WebhookManager:
    """Get global webhook manager instance"""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


# Event emission helpers for automation system
async def emit_action_event(action: OptimizationAction, event_type: WebhookEventType, 
                          additional_data: Optional[Dict[str, Any]] = None):
    """Emit webhook event for optimization action"""
    webhook_manager = get_webhook_manager()
    
    event_data = {
        "action_id": str(action.id),
        "action_type": action.action_type.value,
        "resource_id": action.resource_id,
        "resource_type": action.resource_type,
        "estimated_savings": float(action.estimated_monthly_savings),
        "actual_savings": float(action.actual_savings) if action.actual_savings else None,
        "risk_level": action.risk_level.value,
        "status": action.execution_status.value,
        "policy_id": str(action.policy_id)
    }
    
    if additional_data:
        event_data.update(additional_data)
    
    event = WebhookEvent(
        event_type=event_type,
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        data=event_data,
        resource_id=action.resource_id,
        action_id=str(action.id)
    )
    
    await webhook_manager.emit_event(event)


async def emit_system_event(event_type: WebhookEventType, data: Dict[str, Any], 
                          resource_id: Optional[str] = None, action_id: Optional[str] = None):
    """Emit webhook event for system-level events"""
    webhook_manager = get_webhook_manager()
    
    event = WebhookEvent(
        event_type=event_type,
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        data=data,
        resource_id=resource_id,
        action_id=action_id
    )
    
    await webhook_manager.emit_event(event)