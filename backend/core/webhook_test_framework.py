"""
Integration testing framework for webhook system.
Provides mock endpoints, test utilities, and validation helpers.
"""

import json
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import structlog
from .webhook_manager import (
    WebhookManager, WebhookEndpoint, WebhookEventType, 
    WebhookSecurityType, WebhookEvent
)

logger = structlog.get_logger(__name__)


@dataclass
class MockWebhookRequest:
    """Captured webhook request for testing"""
    method: str
    url: str
    headers: Dict[str, str]
    body: str
    timestamp: datetime
    parsed_body: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Parse JSON body if possible"""
        try:
            self.parsed_body = json.loads(self.body) if self.body else None
        except json.JSONDecodeError:
            self.parsed_body = None


class MockWebhookServer:
    """Mock HTTP server for testing webhook deliveries"""
    
    def __init__(self, port: int = 0, response_status: int = 200, 
                 response_delay: float = 0.0, response_body: str = "OK"):
        self.port = port
        self.response_status = response_status
        self.response_delay = response_delay
        self.response_body = response_body
        self.requests: List[MockWebhookRequest] = []
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.request_handler = self._create_request_handler()
    
    def _create_request_handler(self):
        """Create request handler class with access to server instance"""
        server_instance = self
        
        class WebhookRequestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self._handle_request()
            
            def do_PUT(self):
                self._handle_request()
            
            def do_PATCH(self):
                self._handle_request()
            
            def _handle_request(self):
                # Add delay if configured
                if server_instance.response_delay > 0:
                    time.sleep(server_instance.response_delay)
                
                # Read request body
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else ""
                
                # Capture request
                request = MockWebhookRequest(
                    method=self.command,
                    url=self.path,
                    headers=dict(self.headers),
                    body=body,
                    timestamp=datetime.now()
                )
                server_instance.requests.append(request)
                
                # Send response
                self.send_response(server_instance.response_status)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(server_instance.response_body.encode('utf-8'))
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        return WebhookRequestHandler
    
    def start(self) -> str:
        """Start the mock server and return the URL"""
        if self.is_running:
            return self.get_url()
        
        self.server = HTTPServer(('localhost', self.port), self.request_handler)
        self.port = self.server.server_port  # Get actual port if 0 was specified
        
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.is_running = True
        return self.get_url()
    
    def stop(self):
        """Stop the mock server"""
        if not self.is_running:
            return
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
        
        self.is_running = False
    
    def get_url(self) -> str:
        """Get the server URL"""
        return f"http://localhost:{self.port}"
    
    def clear_requests(self):
        """Clear captured requests"""
        self.requests.clear()
    
    def get_requests(self) -> List[MockWebhookRequest]:
        """Get all captured requests"""
        return self.requests.copy()
    
    def get_last_request(self) -> Optional[MockWebhookRequest]:
        """Get the last captured request"""
        return self.requests[-1] if self.requests else None
    
    def wait_for_requests(self, count: int, timeout: float = 5.0) -> bool:
        """Wait for a specific number of requests"""
        start_time = time.time()
        while len(self.requests) < count and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        return len(self.requests) >= count


class WebhookTestFramework:
    """Comprehensive testing framework for webhook system"""
    
    def __init__(self):
        self.webhook_manager = WebhookManager()
        self.mock_servers: Dict[str, MockWebhookServer] = {}
        self.test_endpoints: Dict[str, WebhookEndpoint] = {}
    
    async def setup(self):
        """Setup the test framework"""
        await self.webhook_manager.start()
    
    async def teardown(self):
        """Cleanup the test framework"""
        await self.webhook_manager.stop()
        
        # Stop all mock servers
        for server in self.mock_servers.values():
            server.stop()
        
        self.mock_servers.clear()
        self.test_endpoints.clear()
    
    def create_mock_server(self, server_id: str, response_status: int = 200, 
                          response_delay: float = 0.0, response_body: str = "OK") -> MockWebhookServer:
        """Create and start a mock webhook server"""
        server = MockWebhookServer(
            response_status=response_status,
            response_delay=response_delay,
            response_body=response_body
        )
        
        server.start()
        self.mock_servers[server_id] = server
        return server
    
    def create_test_endpoint(self, endpoint_id: str, server_id: str, 
                           event_types: List[WebhookEventType],
                           security_type: WebhookSecurityType = WebhookSecurityType.NONE,
                           security_config: Dict[str, Any] = None) -> WebhookEndpoint:
        """Create a test webhook endpoint"""
        if server_id not in self.mock_servers:
            raise ValueError(f"Mock server {server_id} not found")
        
        server = self.mock_servers[server_id]
        endpoint = WebhookEndpoint(
            id=endpoint_id,
            name=f"Test Endpoint {endpoint_id}",
            url=server.get_url() + "/webhook",
            event_types=event_types,
            security_type=security_type,
            security_config=security_config or {},
            timeout=10,
            retry_attempts=1,  # Reduce retries for faster testing
            retry_delay=1
        )
        
        self.webhook_manager.register_endpoint(endpoint)
        self.test_endpoints[endpoint_id] = endpoint
        return endpoint
    
    async def emit_test_event(self, event_type: WebhookEventType, 
                            data: Dict[str, Any] = None) -> WebhookEvent:
        """Emit a test webhook event"""
        event = WebhookEvent(
            event_type=event_type,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            data=data or {"test": True},
            resource_id="test-resource",
            action_id="test-action"
        )
        
        await self.webhook_manager.emit_event(event)
        return event
    
    async def wait_for_deliveries(self, server_id: str, count: int, timeout: float = 5.0) -> bool:
        """Wait for webhook deliveries to a mock server"""
        if server_id not in self.mock_servers:
            return False
        
        server = self.mock_servers[server_id]
        return server.wait_for_requests(count, timeout)
    
    def get_server_requests(self, server_id: str) -> List[MockWebhookRequest]:
        """Get requests received by a mock server"""
        if server_id not in self.mock_servers:
            return []
        
        return self.mock_servers[server_id].get_requests()
    
    def validate_webhook_payload(self, request: MockWebhookRequest, 
                               expected_event_type: WebhookEventType) -> bool:
        """Validate webhook payload structure and content"""
        if not request.parsed_body:
            return False
        
        payload = request.parsed_body
        
        # Check required fields
        required_fields = ['event_id', 'event_type', 'timestamp', 'data', 'version']
        for field in required_fields:
            if field not in payload:
                return False
        
        # Check event type
        if payload['event_type'] != expected_event_type.value:
            return False
        
        # Check timestamp format
        try:
            datetime.fromisoformat(payload['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            return False
        
        return True
    
    def validate_security_headers(self, request: MockWebhookRequest, 
                                security_type: WebhookSecurityType,
                                security_config: Dict[str, Any]) -> bool:
        """Validate security headers in webhook request"""
        headers = request.headers
        
        if security_type == WebhookSecurityType.BEARER_TOKEN:
            auth_header = headers.get('Authorization', '')
            expected_token = security_config.get('token', '')
            return auth_header == f"Bearer {expected_token}"
        
        elif security_type == WebhookSecurityType.HMAC_SHA256:
            signature_header = headers.get('X-Webhook-Signature', '')
            if not signature_header.startswith('sha256='):
                return False
            
            # Validate HMAC signature
            import hmac
            import hashlib
            
            secret = security_config.get('secret', '').encode('utf-8')
            payload = request.body.encode('utf-8')
            expected_signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
            received_signature = signature_header[7:]  # Remove 'sha256=' prefix
            
            return hmac.compare_digest(expected_signature, received_signature)
        
        elif security_type == WebhookSecurityType.BASIC_AUTH:
            auth_header = headers.get('Authorization', '')
            if not auth_header.startswith('Basic '):
                return False
            
            import base64
            username = security_config.get('username', '')
            password = security_config.get('password', '')
            expected_credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            received_credentials = auth_header[6:]  # Remove 'Basic ' prefix
            
            return expected_credentials == received_credentials
        
        return True  # No security or unknown type
    
    async def test_endpoint_connectivity(self, endpoint_id: str) -> Dict[str, Any]:
        """Test connectivity to a webhook endpoint"""
        if endpoint_id not in self.test_endpoints:
            return {"success": False, "error": "Endpoint not found"}
        
        endpoint = self.test_endpoints[endpoint_id]
        
        try:
            # Emit test event
            test_event = await self.emit_test_event(
                WebhookEventType.SYSTEM_ERROR,  # Use a generic event type
                {"test_connectivity": True}
            )
            
            # Wait for delivery
            server_id = None
            for sid, server in self.mock_servers.items():
                if server.get_url() in endpoint.url:
                    server_id = sid
                    break
            
            if not server_id:
                return {"success": False, "error": "Associated mock server not found"}
            
            success = await self.wait_for_deliveries(server_id, 1, timeout=10.0)
            
            if success:
                requests = self.get_server_requests(server_id)
                last_request = requests[-1] if requests else None
                
                return {
                    "success": True,
                    "response_time": (datetime.now() - test_event.timestamp).total_seconds(),
                    "request_received": last_request is not None,
                    "payload_valid": self.validate_webhook_payload(last_request, WebhookEventType.SYSTEM_ERROR) if last_request else False
                }
            else:
                return {"success": False, "error": "No request received within timeout"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_event_filtering(self, endpoint_id: str) -> Dict[str, Any]:
        """Test that endpoints only receive events they're subscribed to"""
        if endpoint_id not in self.test_endpoints:
            return {"success": False, "error": "Endpoint not found"}
        
        endpoint = self.test_endpoints[endpoint_id]
        subscribed_events = endpoint.event_types
        
        # Find associated server
        server_id = None
        for sid, server in self.mock_servers.items():
            if server.get_url() in endpoint.url:
                server_id = sid
                break
        
        if not server_id:
            return {"success": False, "error": "Associated mock server not found"}
        
        server = self.mock_servers[server_id]
        server.clear_requests()
        
        # Emit events of different types
        all_event_types = list(WebhookEventType)
        emitted_events = []
        
        for event_type in all_event_types[:5]:  # Test first 5 event types
            event = await self.emit_test_event(event_type, {"test_filtering": True})
            emitted_events.append(event_type)
        
        # Wait for deliveries
        await asyncio.sleep(2.0)  # Give time for all deliveries
        
        requests = server.get_requests()
        received_event_types = []
        
        for request in requests:
            if request.parsed_body and 'event_type' in request.parsed_body:
                received_event_types.append(WebhookEventType(request.parsed_body['event_type']))
        
        # Check filtering
        expected_events = [et for et in emitted_events if et in subscribed_events]
        unexpected_events = [et for et in received_event_types if et not in subscribed_events]
        
        return {
            "success": len(unexpected_events) == 0,
            "subscribed_events": [e.value for e in subscribed_events],
            "emitted_events": [e.value for e in emitted_events],
            "received_events": [e.value for e in received_event_types],
            "expected_events": [e.value for e in expected_events],
            "unexpected_events": [e.value for e in unexpected_events],
            "filtering_correct": set(received_event_types) == set(expected_events)
        }
    
    async def test_retry_mechanism(self, endpoint_id: str) -> Dict[str, Any]:
        """Test webhook retry mechanism with failing server"""
        if endpoint_id not in self.test_endpoints:
            return {"success": False, "error": "Endpoint not found"}
        
        # Create a failing server
        failing_server = self.create_mock_server(
            f"failing_{endpoint_id}",
            response_status=500,
            response_body="Internal Server Error"
        )
        
        # Update endpoint URL to point to failing server
        endpoint = self.test_endpoints[endpoint_id]
        original_url = endpoint.url
        endpoint.url = failing_server.get_url() + "/webhook"
        endpoint.retry_attempts = 3
        endpoint.retry_delay = 1
        
        try:
            # Emit test event
            test_event = await self.emit_test_event(
                WebhookEventType.SYSTEM_ERROR,
                {"test_retry": True}
            )
            
            # Wait for all retry attempts
            success = failing_server.wait_for_requests(3, timeout=15.0)
            requests = failing_server.get_requests()
            
            return {
                "success": success,
                "retry_attempts_made": len(requests),
                "expected_retries": 3,
                "retry_working": len(requests) == 3,
                "request_timestamps": [r.timestamp.isoformat() for r in requests]
            }
        
        finally:
            # Restore original URL
            endpoint.url = original_url
            failing_server.stop()
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite for webhook system"""
        results = {
            "test_suite_started": datetime.now().isoformat(),
            "tests": {}
        }
        
        try:
            # Test 1: Basic connectivity
            server1 = self.create_mock_server("test_server_1")
            endpoint1 = self.create_test_endpoint(
                "test_endpoint_1",
                "test_server_1",
                [WebhookEventType.ACTION_CREATED, WebhookEventType.ACTION_COMPLETED]
            )
            
            results["tests"]["connectivity"] = await self.test_endpoint_connectivity("test_endpoint_1")
            
            # Test 2: Event filtering
            results["tests"]["event_filtering"] = await self.test_event_filtering("test_endpoint_1")
            
            # Test 3: Security (HMAC)
            server2 = self.create_mock_server("test_server_2")
            endpoint2 = self.create_test_endpoint(
                "test_endpoint_2",
                "test_server_2",
                [WebhookEventType.SYSTEM_ERROR],
                WebhookSecurityType.HMAC_SHA256,
                {"secret": "test_secret_key"}
            )
            
            # Emit event and check security headers
            await self.emit_test_event(WebhookEventType.SYSTEM_ERROR, {"test_security": True})
            await self.wait_for_deliveries("test_server_2", 1)
            
            security_requests = self.get_server_requests("test_server_2")
            security_valid = False
            if security_requests:
                security_valid = self.validate_security_headers(
                    security_requests[-1],
                    WebhookSecurityType.HMAC_SHA256,
                    {"secret": "test_secret_key"}
                )
            
            results["tests"]["security"] = {
                "success": security_valid,
                "requests_received": len(security_requests),
                "security_headers_valid": security_valid
            }
            
            # Test 4: Retry mechanism
            results["tests"]["retry_mechanism"] = await self.test_retry_mechanism("test_endpoint_1")
            
            # Test 5: Multiple endpoints
            server3 = self.create_mock_server("test_server_3")
            endpoint3 = self.create_test_endpoint(
                "test_endpoint_3",
                "test_server_3",
                [WebhookEventType.ACTION_FAILED]
            )
            
            # Clear all servers
            for server in self.mock_servers.values():
                server.clear_requests()
            
            # Emit event that should go to multiple endpoints
            await self.emit_test_event(WebhookEventType.ACTION_CREATED, {"test_multiple": True})
            await asyncio.sleep(2.0)
            
            server1_requests = len(self.get_server_requests("test_server_1"))
            server2_requests = len(self.get_server_requests("test_server_2"))
            server3_requests = len(self.get_server_requests("test_server_3"))
            
            results["tests"]["multiple_endpoints"] = {
                "success": server1_requests > 0 and server2_requests == 0 and server3_requests == 0,
                "server1_requests": server1_requests,
                "server2_requests": server2_requests,
                "server3_requests": server3_requests,
                "filtering_correct": server1_requests > 0 and server2_requests == 0 and server3_requests == 0
            }
            
        except Exception as e:
            results["error"] = str(e)
        
        results["test_suite_completed"] = datetime.now().isoformat()
        results["overall_success"] = all(
            test_result.get("success", False) 
            for test_result in results["tests"].values()
        )
        
        return results


# Utility functions for testing
async def create_test_webhook_system() -> WebhookTestFramework:
    """Create and setup a test webhook system"""
    framework = WebhookTestFramework()
    await framework.setup()
    return framework


async def run_webhook_integration_tests() -> Dict[str, Any]:
    """Run complete webhook integration test suite"""
    framework = await create_test_webhook_system()
    
    try:
        return await framework.run_comprehensive_test_suite()
    finally:
        await framework.teardown()