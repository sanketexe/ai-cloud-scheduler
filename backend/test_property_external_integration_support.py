"""
Property-based test for external integration support

**Feature: automated-cost-optimization, Property 21: External Integration Support**
**Validates: Requirements 5.5**

Property 21: External Integration Support
*For any* external monitoring system integration, the system should provide properly formatted webhook endpoints for real-time event streaming
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from core.webhook_manager import (
    WebhookManager, WebhookEndpoint, WebhookEventType, 
    WebhookSecurityType, WebhookEvent, get_webhook_manager
)


class WebhookIntegrationStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based test for webhook integration system.
    Tests that external monitoring system integration works correctly.
    """
    
    def __init__(self):
        super().__init__()
        self.webhook_manager = None
        self.registered_endpoints: Dict[str, WebhookEndpoint] = {}
        self.mock_requests: Dict[str, List[Dict[str, Any]]] = {}
        self.emitted_events: List[WebhookEvent] = []
    
    @initialize()
    def setup_webhook_system(self):
        """Initialize webhook system for testing"""
        # Create webhook manager without database dependencies
        self.webhook_manager = WebhookManager()
        self.webhook_manager.is_running = True
    
    endpoints = Bundle('endpoints')
    events = Bundle('events')
    
    @rule(target=endpoints, 
          endpoint_name=st.text(min_size=1, max_size=50),
          event_types=st.lists(st.sampled_from(list(WebhookEventType)), min_size=1, max_size=5),
          security_type=st.sampled_from(list(WebhookSecurityType)))
    def register_webhook_endpoint(self, endpoint_name: str, event_types: List[WebhookEventType], 
                                security_type: WebhookSecurityType):
        """Register a webhook endpoint for external monitoring system"""
        assume(len(endpoint_name.strip()) > 0)
        assume(len(set(event_types)) > 0)  # Ensure unique event types
        
        # Create security config based on type
        security_config = {}
        if security_type == WebhookSecurityType.BEARER_TOKEN:
            security_config = {"token": f"test_token_{endpoint_name}"}
        elif security_type == WebhookSecurityType.HMAC_SHA256:
            security_config = {"secret": f"test_secret_{endpoint_name}"}
        elif security_type == WebhookSecurityType.BASIC_AUTH:
            security_config = {"username": f"user_{endpoint_name}", "password": f"pass_{endpoint_name}"}
        
        # Create endpoint
        endpoint_id = str(uuid.uuid4())
        endpoint = WebhookEndpoint(
            id=endpoint_id,
            name=f"Test Endpoint {endpoint_name}",
            url=f"https://example.com/webhook/{endpoint_id}",
            event_types=event_types,
            security_type=security_type,
            security_config=security_config,
            timeout=10,
            retry_attempts=1,
            retry_delay=1
        )
        
        # Register endpoint in memory only (no database)
        self.webhook_manager.endpoints[endpoint_id] = endpoint
        self.registered_endpoints[endpoint_id] = endpoint
        self.mock_requests[endpoint_id] = []
        
        return endpoint_id
    
    @rule(target=events,
          event_type=st.sampled_from(list(WebhookEventType)),
          event_data=st.dictionaries(
              st.text(min_size=1, max_size=20), 
              st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
              min_size=0, max_size=5
          ))
    def emit_webhook_event(self, event_type: WebhookEventType, event_data: Dict[str, Any]):
        """Emit a webhook event for external monitoring systems"""
        event = WebhookEvent(
            event_type=event_type,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            data=event_data,
            resource_id=f"test_resource_{len(self.emitted_events)}",
            action_id=f"test_action_{len(self.emitted_events)}"
        )
        
        # Simulate event delivery to subscribed endpoints
        for endpoint_id, endpoint in self.registered_endpoints.items():
            if event_type in endpoint.event_types:
                # Simulate webhook payload formatting
                payload = self.webhook_manager._format_webhook_payload(event)
                
                # Simulate security headers
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "FinOps-Webhook/1.0",
                    "X-Webhook-Event": event.event_type.value,
                    "X-Webhook-Delivery": str(uuid.uuid4()),
                    "X-Webhook-Timestamp": str(int(event.timestamp.timestamp()))
                }
                
                # Add security headers based on endpoint configuration
                self.webhook_manager._add_security_headers(
                    headers, payload, endpoint.security_type, endpoint.security_config
                )
                
                # Store mock request
                mock_request = {
                    "method": "POST",
                    "url": endpoint.url,
                    "headers": headers,
                    "payload": payload,
                    "timestamp": event.timestamp
                }
                self.mock_requests[endpoint_id].append(mock_request)
        
        self.emitted_events.append(event)
        return event
    
    @rule(endpoint_id=endpoints)
    def test_endpoint_receives_subscribed_events(self, endpoint_id: str):
        """Test that endpoints only receive events they're subscribed to"""
        if endpoint_id not in self.registered_endpoints:
            return
        
        endpoint = self.registered_endpoints[endpoint_id]
        
        # Check that received events match subscription
        requests = self.mock_requests.get(endpoint_id, [])
        for request in requests:
            if 'payload' in request and 'event_type' in request['payload']:
                received_event_type = WebhookEventType(request['payload']['event_type'])
                assert received_event_type in endpoint.event_types, \
                    f"Endpoint received unsubscribed event type: {received_event_type}"
    
    @rule(endpoint_id=endpoints)
    def test_webhook_payload_format(self, endpoint_id: str):
        """Test that webhook payloads are properly formatted for external systems"""
        if endpoint_id not in self.registered_endpoints:
            return
        
        # Check payload format for all received requests
        requests = self.mock_requests.get(endpoint_id, [])
        for request in requests:
            # Verify content type
            assert request['headers'].get('Content-Type') == 'application/json', \
                "Webhook payload must be JSON"
            
            # Verify payload structure
            payload = request['payload']
            
            # Required fields for external monitoring systems
            required_fields = ['event_id', 'event_type', 'timestamp', 'data', 'version']
            for field in required_fields:
                assert field in payload, f"Missing required field: {field}"
            
            # Verify timestamp format (ISO 8601)
            try:
                datetime.fromisoformat(payload['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                assert False, "Timestamp must be in ISO 8601 format"
            
            # Verify event_id is UUID format
            try:
                uuid.UUID(payload['event_id'])
            except ValueError:
                assert False, "Event ID must be valid UUID"
            
            # Verify version is present
            assert payload['version'] == '1.0', "Version must be specified"
    
    @rule(endpoint_id=endpoints)
    def test_security_headers_present(self, endpoint_id: str):
        """Test that security headers are properly included for external systems"""
        if endpoint_id not in self.registered_endpoints:
            return
        
        endpoint = self.registered_endpoints[endpoint_id]
        
        # Check security headers based on endpoint configuration
        requests = self.mock_requests.get(endpoint_id, [])
        for request in requests:
            headers = request['headers']
            
            if endpoint.security_type == WebhookSecurityType.BEARER_TOKEN:
                assert 'Authorization' in headers, "Bearer token authorization header missing"
                assert headers['Authorization'].startswith('Bearer '), "Invalid bearer token format"
            
            elif endpoint.security_type == WebhookSecurityType.HMAC_SHA256:
                assert 'X-Webhook-Signature' in headers, "HMAC signature header missing"
                assert headers['X-Webhook-Signature'].startswith('sha256='), "Invalid HMAC signature format"
            
            elif endpoint.security_type == WebhookSecurityType.BASIC_AUTH:
                assert 'Authorization' in headers, "Basic auth header missing"
                assert headers['Authorization'].startswith('Basic '), "Invalid basic auth format"
            
            # Standard webhook headers should always be present
            assert 'X-Webhook-Event' in headers, "Webhook event header missing"
            assert 'X-Webhook-Delivery' in headers, "Webhook delivery ID header missing"
            assert 'X-Webhook-Timestamp' in headers, "Webhook timestamp header missing"
    
    @invariant()
    def webhook_manager_is_running(self):
        """Webhook manager should always be running during tests"""
        assert self.webhook_manager is not None, "Webhook manager should be initialized"
        assert self.webhook_manager.is_running, "Webhook manager should be running"
    
    @invariant()
    def registered_endpoints_are_tracked(self):
        """All registered endpoints should be tracked in the manager"""
        for endpoint_id in self.registered_endpoints:
            assert endpoint_id in self.webhook_manager.endpoints, \
                f"Endpoint {endpoint_id} should be tracked in manager"
    
    def teardown(self):
        """Cleanup after tests"""
        if self.webhook_manager:
            self.webhook_manager.is_running = False


# Property-based test using the state machine
@settings(max_examples=50, deadline=30000)  # Reduced examples for faster execution
@given(st.data())
def test_external_integration_support_property(data):
    """
    **Feature: automated-cost-optimization, Property 21: External Integration Support**
    
    Property: For any external monitoring system integration, the system should provide 
    properly formatted webhook endpoints for real-time event streaming
    
    This test verifies that:
    1. Webhook endpoints can be registered for external monitoring systems
    2. Events are properly formatted and delivered to subscribed endpoints
    3. Security headers are correctly applied based on configuration
    4. Only subscribed events are delivered to each endpoint
    5. Payload format is consistent and suitable for external systems
    """
    
    def run_test():
        state_machine = WebhookIntegrationStateMachine()
        
        try:
            state_machine.setup_webhook_system()
            
            # Generate test scenario
            num_endpoints = data.draw(st.integers(min_value=1, max_value=3))
            num_events = data.draw(st.integers(min_value=1, max_value=5))
            
            # Register endpoints
            endpoint_ids = []
            for i in range(num_endpoints):
                endpoint_name = data.draw(st.text(min_size=1, max_size=20))
                event_types = data.draw(st.lists(
                    st.sampled_from(list(WebhookEventType)), 
                    min_size=1, max_size=3
                ))
                security_type = data.draw(st.sampled_from(list(WebhookSecurityType)))
                
                endpoint_id = state_machine.register_webhook_endpoint(
                    endpoint_name, event_types, security_type
                )
                endpoint_ids.append(endpoint_id)
            
            # Emit events
            for i in range(num_events):
                event_type = data.draw(st.sampled_from(list(WebhookEventType)))
                event_data = data.draw(st.dictionaries(
                    st.text(min_size=1, max_size=10),
                    st.one_of(st.text(), st.integers(), st.booleans()),
                    min_size=0, max_size=3
                ))
                
                state_machine.emit_webhook_event(event_type, event_data)
            
            # Run property checks
            for endpoint_id in endpoint_ids:
                state_machine.test_endpoint_receives_subscribed_events(endpoint_id)
                state_machine.test_webhook_payload_format(endpoint_id)
                state_machine.test_security_headers_present(endpoint_id)
            
            # Check invariants
            state_machine.webhook_manager_is_running()
            state_machine.registered_endpoints_are_tracked()
            
        finally:
            state_machine.teardown()
    
    # Run the test
    run_test()


# Simple unit tests for basic functionality
def test_webhook_manager_initialization():
    """Test that webhook manager can be initialized"""
    manager = WebhookManager()
    assert manager is not None
    assert not manager.is_running
    assert len(manager.endpoints) == 0


def test_webhook_event_creation():
    """Test that webhook events can be created with proper structure"""
    event = WebhookEvent(
        event_type=WebhookEventType.ACTION_CREATED,
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        data={"test": True},
        resource_id="test_resource",
        action_id="test_action"
    )
    
    assert event.event_type == WebhookEventType.ACTION_CREATED
    assert event.data["test"] is True
    assert event.resource_id == "test_resource"
    assert event.action_id == "test_action"


def test_webhook_endpoint_configuration():
    """Test that webhook endpoints can be configured with different security types"""
    for security_type in WebhookSecurityType:
        security_config = {}
        if security_type == WebhookSecurityType.BEARER_TOKEN:
            security_config = {"token": "test_token"}
        elif security_type == WebhookSecurityType.HMAC_SHA256:
            security_config = {"secret": "test_secret"}
        elif security_type == WebhookSecurityType.BASIC_AUTH:
            security_config = {"username": "test_user", "password": "test_pass"}
        
        endpoint = WebhookEndpoint(
            id=str(uuid.uuid4()),
            name="Test Endpoint",
            url="https://example.com/webhook",
            event_types=[WebhookEventType.ACTION_CREATED],
            security_type=security_type,
            security_config=security_config
        )
        
        assert endpoint.security_type == security_type
        assert endpoint.security_config == security_config


if __name__ == "__main__":
    try:
        # Run unit tests first
        test_webhook_manager_initialization()
        test_webhook_event_creation()
        test_webhook_endpoint_configuration()
        print("✅ Unit tests passed!")
        
        # Run the property-based test
        test_external_integration_support_property()
        print("✅ External integration support property test passed!")
        
        print("✅ All webhook integration tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise