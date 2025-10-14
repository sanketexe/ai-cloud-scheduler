# api_documentation.py
"""
API Documentation and Testing Tools

This module provides comprehensive API documentation with interactive examples
and automated testing capabilities:
- OpenAPI/Swagger documentation generation
- Interactive API explorer
- Automated API testing suite
- API versioning and backward compatibility management

Requirements addressed:
- 5.6: Comprehensive API documentation and testing
"""

import json
import yaml
import requests
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from enum import Enum
import inspect
from functools import wraps
import asyncio
import time
from urllib.parse import urljoin
import hashlib


class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class APIParameter:
    """API parameter definition"""
    name: str
    param_type: str  # query, path, header, body
    data_type: str  # string, integer, boolean, object, array
    required: bool = False
    description: str = ""
    example: Any = None
    enum_values: Optional[List[str]] = None
    format: Optional[str] = None  # date, date-time, email, etc.


@dataclass
class APIResponse:
    """API response definition"""
    status_code: int
    description: str
    schema: Optional[Dict[str, Any]] = None
    example: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class APIEndpoint:
    """API endpoint definition"""
    path: str
    method: HTTPMethod
    summary: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[APIParameter] = field(default_factory=list)
    responses: List[APIResponse] = field(default_factory=list)
    deprecated: bool = False
    version: APIVersion = APIVersion.V1
    
    def to_openapi_spec(self) -> Dict[str, Any]:
        """Convert to OpenAPI specification format"""
        spec = {
            "summary": self.summary,
            "description": self.description,
            "tags": self.tags,
            "deprecated": self.deprecated,
            "parameters": [],
            "responses": {}
        }
        
        # Add parameters
        for param in self.parameters:
            param_spec = {
                "name": param.name,
                "in": param.param_type,
                "required": param.required,
                "description": param.description,
                "schema": {
                    "type": param.data_type
                }
            }
            
            if param.example is not None:
                param_spec["example"] = param.example
                
            if param.enum_values:
                param_spec["schema"]["enum"] = param.enum_values
                
            if param.format:
                param_spec["schema"]["format"] = param.format
                
            spec["parameters"].append(param_spec)
            
        # Add responses
        for response in self.responses:
            response_spec = {
                "description": response.description
            }
            
            if response.schema:
                response_spec["content"] = {
                    "application/json": {
                        "schema": response.schema
                    }
                }
                
            if response.example:
                if "content" not in response_spec:
                    response_spec["content"] = {"application/json": {}}
                response_spec["content"]["application/json"]["example"] = response.example
                
            spec["responses"][str(response.status_code)] = response_spec
            
        return spec


@dataclass
class APITestCase:
    """API test case definition"""
    test_id: str
    name: str
    description: str
    endpoint: APIEndpoint
    test_data: Dict[str, Any] = field(default_factory=dict)
    expected_status: int = 200
    expected_response: Optional[Dict[str, Any]] = None
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    timeout: int = 30
    
    
@dataclass
class APITestResult:
    """API test result"""
    test_id: str
    status: TestStatus
    execution_time: float
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class APIDocumentationGenerator:
    """Generates comprehensive API documentation"""
    
    def __init__(self, title: str = "Cloud Intelligence Platform API", version: str = "1.0.0"):
        self.title = title
        self.version = version
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.logger = logging.getLogger(__name__)
        
    def add_endpoint(self, endpoint: APIEndpoint) -> None:
        """Add an API endpoint to documentation"""
        endpoint_key = f"{endpoint.method.value}:{endpoint.path}"
        self.endpoints[endpoint_key] = endpoint
        self.logger.info(f"Added endpoint to documentation: {endpoint_key}")
        
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification"""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": "Comprehensive API for the Cloud Intelligence Platform"
            },
            "servers": [
                {
                    "url": "https://api.cloudintelligence.com/v1",
                    "description": "Production server"
                },
                {
                    "url": "https://staging-api.cloudintelligence.com/v1",
                    "description": "Staging server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": self._generate_schemas(),
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    },
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                }
            },
            "security": [
                {"ApiKeyAuth": []},
                {"BearerAuth": []}
            ]
        }
        
        # Group endpoints by path
        paths = {}
        for endpoint in self.endpoints.values():
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            paths[endpoint.path][endpoint.method.value.lower()] = endpoint.to_openapi_spec()
            
        spec["paths"] = paths
        return spec
        
    def _generate_schemas(self) -> Dict[str, Any]:
        """Generate common schema definitions"""
        return {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "description": "Error message"
                    },
                    "code": {
                        "type": "integer",
                        "description": "Error code"
                    },
                    "details": {
                        "type": "object",
                        "description": "Additional error details"
                    }
                },
                "required": ["error", "code"]
            },
            "Workload": {
                "type": "object",
                "properties": {
                    "workload_id": {"type": "string"},
                    "name": {"type": "string"},
                    "cpu_request": {"type": "number"},
                    "memory_request": {"type": "number"},
                    "priority": {"type": "integer"},
                    "cost_constraints": {"$ref": "#/components/schemas/CostConstraints"},
                    "performance_requirements": {"$ref": "#/components/schemas/PerformanceRequirements"}
                },
                "required": ["workload_id", "name", "cpu_request", "memory_request"]
            },
            "CostConstraints": {
                "type": "object",
                "properties": {
                    "max_hourly_cost": {"type": "number"},
                    "max_monthly_budget": {"type": "number"},
                    "cost_optimization_preference": {"type": "string", "enum": ["low", "medium", "high"]}
                }
            },
            "PerformanceRequirements": {
                "type": "object",
                "properties": {
                    "min_cpu_performance": {"type": "number"},
                    "max_latency_ms": {"type": "integer"},
                    "availability_requirement": {"type": "number"}
                }
            },
            "VirtualMachine": {
                "type": "object",
                "properties": {
                    "vm_id": {"type": "string"},
                    "provider": {"type": "string"},
                    "instance_type": {"type": "string"},
                    "cpu_cores": {"type": "integer"},
                    "memory_gb": {"type": "number"},
                    "hourly_cost": {"type": "number"},
                    "availability_zone": {"type": "string"},
                    "status": {"type": "string", "enum": ["available", "busy", "maintenance"]}
                },
                "required": ["vm_id", "provider", "instance_type", "cpu_cores", "memory_gb"]
            },
            "Budget": {
                "type": "object",
                "properties": {
                    "budget_id": {"type": "string"},
                    "name": {"type": "string"},
                    "amount": {"type": "number"},
                    "period": {"type": "string", "enum": ["monthly", "quarterly", "yearly"]},
                    "alert_thresholds": {"type": "array", "items": {"type": "number"}},
                    "current_spend": {"type": "number"}
                },
                "required": ["budget_id", "name", "amount", "period"]
            },
            "PerformanceMetrics": {
                "type": "object",
                "properties": {
                    "resource_id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "cpu_utilization": {"type": "number"},
                    "memory_utilization": {"type": "number"},
                    "network_io": {"type": "number"},
                    "disk_io": {"type": "number"}
                },
                "required": ["resource_id", "timestamp"]
            },
            "SimulationResult": {
                "type": "object",
                "properties": {
                    "simulation_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["running", "completed", "failed"]},
                    "total_cost": {"type": "number"},
                    "resource_efficiency": {"type": "number"},
                    "sla_compliance": {"type": "number"},
                    "recommendations": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["simulation_id", "status"]
            }
        }
        
    def export_to_file(self, filename: str, format: str = "yaml") -> None:
        """Export documentation to file"""
        spec = self.generate_openapi_spec()
        
        if format.lower() == "yaml":
            with open(filename, 'w') as f:
                yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        elif format.lower() == "json":
            with open(filename, 'w') as f:
                json.dump(spec, f, indent=2)
        else:
            raise ValueError("Format must be 'yaml' or 'json'")
            
        self.logger.info(f"API documentation exported to {filename}")
        
    def generate_html_docs(self) -> str:
        """Generate HTML documentation using Swagger UI"""
        spec = self.generate_openapi_spec()
        spec_json = json.dumps(spec, indent=2)
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui.css" />
    <style>
        html {{
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }}
        *, *:before, *:after {{
            box-sizing: inherit;
        }}
        body {{
            margin:0;
            background: #fafafa;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                spec: {spec_json},
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            }});
        }};
    </script>
</body>
</html>
        """
        
        return html_template


class APITestSuite:
    """Automated API testing suite"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.test_cases: List[APITestCase] = []
        self.test_results: List[APITestResult] = []
        self.logger = logging.getLogger(__name__)
        
    def add_test_case(self, test_case: APITestCase) -> None:
        """Add a test case to the suite"""
        self.test_cases.append(test_case)
        self.logger.info(f"Added test case: {test_case.name}")
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases"""
        self.test_results = []
        start_time = time.time()
        
        passed = 0
        failed = 0
        errors = 0
        
        for test_case in self.test_cases:
            result = await self._run_single_test(test_case)
            self.test_results.append(result)
            
            if result.status == TestStatus.PASSED:
                passed += 1
            elif result.status == TestStatus.FAILED:
                failed += 1
            elif result.status == TestStatus.ERROR:
                errors += 1
                
        total_time = time.time() - start_time
        
        summary = {
            "total_tests": len(self.test_cases),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "execution_time": total_time,
            "success_rate": (passed / len(self.test_cases)) * 100 if self.test_cases else 0
        }
        
        self.logger.info(f"Test suite completed: {summary}")
        return summary
        
    async def _run_single_test(self, test_case: APITestCase) -> APITestResult:
        """Run a single test case"""
        start_time = time.time()
        
        try:
            # Setup
            if test_case.setup_function:
                await test_case.setup_function()
                
            # Prepare request
            url = urljoin(self.base_url, test_case.endpoint.path)
            headers = {"Content-Type": "application/json"}
            
            if self.api_key:
                headers["X-API-Key"] = self.api_key
                
            # Replace path parameters
            for param in test_case.endpoint.parameters:
                if param.param_type == "path" and param.name in test_case.test_data:
                    url = url.replace(f"{{{param.name}}}", str(test_case.test_data[param.name]))
                    
            # Prepare query parameters
            query_params = {}
            for param in test_case.endpoint.parameters:
                if param.param_type == "query" and param.name in test_case.test_data:
                    query_params[param.name] = test_case.test_data[param.name]
                    
            # Prepare body data
            body_data = {}
            for param in test_case.endpoint.parameters:
                if param.param_type == "body" and param.name in test_case.test_data:
                    body_data[param.name] = test_case.test_data[param.name]
                    
            # Make request
            method = test_case.endpoint.method.value
            
            if method == "GET":
                response = requests.get(url, headers=headers, params=query_params, timeout=test_case.timeout)
            elif method == "POST":
                response = requests.post(url, headers=headers, params=query_params, json=body_data, timeout=test_case.timeout)
            elif method == "PUT":
                response = requests.put(url, headers=headers, params=query_params, json=body_data, timeout=test_case.timeout)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=query_params, timeout=test_case.timeout)
            elif method == "PATCH":
                response = requests.patch(url, headers=headers, params=query_params, json=body_data, timeout=test_case.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            execution_time = time.time() - start_time
            
            # Validate response
            if response.status_code != test_case.expected_status:
                return APITestResult(
                    test_id=test_case.test_id,
                    status=TestStatus.FAILED,
                    execution_time=execution_time,
                    request_data=test_case.test_data,
                    response_data={"status_code": response.status_code, "body": response.text},
                    error_message=f"Expected status {test_case.expected_status}, got {response.status_code}"
                )
                
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = {"body": response.text}
                
            # Validate expected response if provided
            if test_case.expected_response:
                if not self._validate_response_structure(response_data, test_case.expected_response):
                    return APITestResult(
                        test_id=test_case.test_id,
                        status=TestStatus.FAILED,
                        execution_time=execution_time,
                        request_data=test_case.test_data,
                        response_data=response_data,
                        error_message="Response structure doesn't match expected format"
                    )
                    
            # Teardown
            if test_case.teardown_function:
                await test_case.teardown_function()
                
            return APITestResult(
                test_id=test_case.test_id,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                request_data=test_case.test_data,
                response_data=response_data
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return APITestResult(
                test_id=test_case.test_id,
                status=TestStatus.ERROR,
                execution_time=execution_time,
                request_data=test_case.test_data,
                error_message=str(e)
            )
            
    def _validate_response_structure(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Validate response structure matches expected format"""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
                
            if isinstance(expected_value, dict) and isinstance(actual[key], dict):
                if not self._validate_response_structure(actual[key], expected_value):
                    return False
            elif isinstance(expected_value, list) and isinstance(actual[key], list):
                if expected_value and actual[key]:
                    if isinstance(expected_value[0], dict) and isinstance(actual[key][0], dict):
                        if not self._validate_response_structure(actual[key][0], expected_value[0]):
                            return False
                            
        return True
        
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results available"}
            
        passed_tests = [r for r in self.test_results if r.status == TestStatus.PASSED]
        failed_tests = [r for r in self.test_results if r.status == TestStatus.FAILED]
        error_tests = [r for r in self.test_results if r.status == TestStatus.ERROR]
        
        avg_execution_time = sum(r.execution_time for r in self.test_results) / len(self.test_results)
        
        report = {
            "summary": {
                "total_tests": len(self.test_results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "errors": len(error_tests),
                "success_rate": (len(passed_tests) / len(self.test_results)) * 100,
                "average_execution_time": avg_execution_time
            },
            "failed_tests": [
                {
                    "test_id": r.test_id,
                    "error_message": r.error_message,
                    "execution_time": r.execution_time
                }
                for r in failed_tests
            ],
            "error_tests": [
                {
                    "test_id": r.test_id,
                    "error_message": r.error_message,
                    "execution_time": r.execution_time
                }
                for r in error_tests
            ],
            "performance_metrics": {
                "fastest_test": min(self.test_results, key=lambda r: r.execution_time).execution_time,
                "slowest_test": max(self.test_results, key=lambda r: r.execution_time).execution_time,
                "average_time": avg_execution_time
            }
        }
        
        return report


class APIVersionManager:
    """Manages API versioning and backward compatibility"""
    
    def __init__(self):
        self.versions: Dict[APIVersion, APIDocumentationGenerator] = {}
        self.compatibility_matrix: Dict[str, List[APIVersion]] = {}
        self.logger = logging.getLogger(__name__)
        
    def add_version(self, version: APIVersion, documentation: APIDocumentationGenerator) -> None:
        """Add a new API version"""
        self.versions[version] = documentation
        self.logger.info(f"Added API version: {version.value}")
        
    def check_compatibility(self, endpoint_path: str, from_version: APIVersion, to_version: APIVersion) -> bool:
        """Check if an endpoint is compatible between versions"""
        if from_version not in self.versions or to_version not in self.versions:
            return False
            
        from_doc = self.versions[from_version]
        to_doc = self.versions[to_version]
        
        # Find endpoints in both versions
        from_endpoints = [ep for ep in from_doc.endpoints.values() if ep.path == endpoint_path]
        to_endpoints = [ep for ep in to_doc.endpoints.values() if ep.path == endpoint_path]
        
        if not from_endpoints or not to_endpoints:
            return False
            
        # Check parameter compatibility
        from_endpoint = from_endpoints[0]
        to_endpoint = to_endpoints[0]
        
        # Required parameters in new version should be subset of old version
        from_required = {p.name for p in from_endpoint.parameters if p.required}
        to_required = {p.name for p in to_endpoint.parameters if p.required}
        
        # New version shouldn't add required parameters
        if not to_required.issubset(from_required):
            return False
            
        return True
        
    def generate_compatibility_report(self) -> Dict[str, Any]:
        """Generate compatibility report between versions"""
        report = {
            "versions": [v.value for v in self.versions.keys()],
            "compatibility_matrix": {},
            "breaking_changes": []
        }
        
        versions = list(self.versions.keys())
        for i, from_version in enumerate(versions):
            for j, to_version in enumerate(versions):
                if i != j:
                    key = f"{from_version.value}_to_{to_version.value}"
                    report["compatibility_matrix"][key] = self._check_version_compatibility(from_version, to_version)
                    
        return report
        
    def _check_version_compatibility(self, from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
        """Check compatibility between two versions"""
        from_doc = self.versions[from_version]
        to_doc = self.versions[to_version]
        
        compatible_endpoints = []
        incompatible_endpoints = []
        
        # Get all unique paths
        all_paths = set()
        all_paths.update(ep.path for ep in from_doc.endpoints.values())
        all_paths.update(ep.path for ep in to_doc.endpoints.values())
        
        for path in all_paths:
            if self.check_compatibility(path, from_version, to_version):
                compatible_endpoints.append(path)
            else:
                incompatible_endpoints.append(path)
                
        return {
            "compatible_endpoints": compatible_endpoints,
            "incompatible_endpoints": incompatible_endpoints,
            "compatibility_score": len(compatible_endpoints) / len(all_paths) * 100 if all_paths else 100
        }


# Convenience functions for creating common API endpoints

def create_workload_endpoints() -> List[APIEndpoint]:
    """Create workload management endpoints"""
    endpoints = []
    
    # List workloads
    endpoints.append(APIEndpoint(
        path="/workloads",
        method=HTTPMethod.GET,
        summary="List all workloads",
        description="Retrieve a list of all workloads with optional filtering",
        tags=["workloads"],
        parameters=[
            APIParameter("limit", "query", "integer", False, "Maximum number of results", 100),
            APIParameter("offset", "query", "integer", False, "Number of results to skip", 0),
            APIParameter("status", "query", "string", False, "Filter by status", enum_values=["pending", "running", "completed"])
        ],
        responses=[
            APIResponse(200, "Successful response", example={"workloads": [], "total": 0}),
            APIResponse(400, "Bad request"),
            APIResponse(401, "Unauthorized")
        ]
    ))
    
    # Create workload
    endpoints.append(APIEndpoint(
        path="/workloads",
        method=HTTPMethod.POST,
        summary="Create a new workload",
        description="Create a new workload for scheduling",
        tags=["workloads"],
        parameters=[
            APIParameter("workload", "body", "object", True, "Workload configuration")
        ],
        responses=[
            APIResponse(201, "Workload created successfully"),
            APIResponse(400, "Invalid workload configuration"),
            APIResponse(401, "Unauthorized")
        ]
    ))
    
    # Get workload by ID
    endpoints.append(APIEndpoint(
        path="/workloads/{workload_id}",
        method=HTTPMethod.GET,
        summary="Get workload by ID",
        description="Retrieve detailed information about a specific workload",
        tags=["workloads"],
        parameters=[
            APIParameter("workload_id", "path", "string", True, "Workload identifier")
        ],
        responses=[
            APIResponse(200, "Workload details"),
            APIResponse(404, "Workload not found"),
            APIResponse(401, "Unauthorized")
        ]
    ))
    
    return endpoints


def create_cost_management_endpoints() -> List[APIEndpoint]:
    """Create cost management endpoints"""
    endpoints = []
    
    # Get cost summary
    endpoints.append(APIEndpoint(
        path="/costs/summary",
        method=HTTPMethod.GET,
        summary="Get cost summary",
        description="Retrieve cost summary for specified time period",
        tags=["costs"],
        parameters=[
            APIParameter("start_date", "query", "string", True, "Start date (ISO format)", format="date"),
            APIParameter("end_date", "query", "string", True, "End date (ISO format)", format="date"),
            APIParameter("provider", "query", "string", False, "Filter by cloud provider")
        ],
        responses=[
            APIResponse(200, "Cost summary", example={"total_cost": 1234.56, "breakdown": {}}),
            APIResponse(400, "Invalid date range"),
            APIResponse(401, "Unauthorized")
        ]
    ))
    
    # Create budget
    endpoints.append(APIEndpoint(
        path="/budgets",
        method=HTTPMethod.POST,
        summary="Create a new budget",
        description="Create a new budget with alerts and thresholds",
        tags=["budgets"],
        parameters=[
            APIParameter("budget", "body", "object", True, "Budget configuration")
        ],
        responses=[
            APIResponse(201, "Budget created successfully"),
            APIResponse(400, "Invalid budget configuration"),
            APIResponse(401, "Unauthorized")
        ]
    ))
    
    return endpoints


def create_performance_monitoring_endpoints() -> List[APIEndpoint]:
    """Create performance monitoring endpoints"""
    endpoints = []
    
    # Get performance metrics
    endpoints.append(APIEndpoint(
        path="/metrics/{resource_id}",
        method=HTTPMethod.GET,
        summary="Get performance metrics",
        description="Retrieve performance metrics for a specific resource",
        tags=["monitoring"],
        parameters=[
            APIParameter("resource_id", "path", "string", True, "Resource identifier"),
            APIParameter("start_time", "query", "string", False, "Start time (ISO format)", format="date-time"),
            APIParameter("end_time", "query", "string", False, "End time (ISO format)", format="date-time"),
            APIParameter("metric_type", "query", "string", False, "Type of metric", enum_values=["cpu", "memory", "network", "disk"])
        ],
        responses=[
            APIResponse(200, "Performance metrics", example={"metrics": [], "resource_id": ""}),
            APIResponse(404, "Resource not found"),
            APIResponse(401, "Unauthorized")
        ]
    ))
    
    return endpoints