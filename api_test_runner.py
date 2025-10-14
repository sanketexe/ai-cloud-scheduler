# api_test_runner.py
"""
API Test Runner

Comprehensive test runner for the Cloud Intelligence Platform API.
Includes automated test generation, execution, and reporting.
"""

import asyncio
import json
import sys
from datetime import datetime
from api_documentation import (
    APIDocumentationGenerator, APITestSuite, APIVersionManager,
    create_workload_endpoints, create_cost_management_endpoints,
    create_performance_monitoring_endpoints, APITestCase, APIVersion
)


async def main():
    """Main test runner function"""
    print("Cloud Intelligence Platform API Test Suite")
    print("=" * 50)
    
    # Initialize documentation generator
    doc_generator = APIDocumentationGenerator(
        title="Cloud Intelligence Platform API",
        version="1.0.0"
    )
    
    # Add all endpoints
    workload_endpoints = create_workload_endpoints()
    cost_endpoints = create_cost_management_endpoints()
    monitoring_endpoints = create_performance_monitoring_endpoints()
    
    all_endpoints = workload_endpoints + cost_endpoints + monitoring_endpoints
    
    for endpoint in all_endpoints:
        doc_generator.add_endpoint(endpoint)
    
    print(f"Added {len(all_endpoints)} endpoints to documentation")
    
    # Generate OpenAPI specification
    openapi_spec = doc_generator.generate_openapi_spec()
    
    # Export documentation
    doc_generator.export_to_file("api_spec.yaml", "yaml")
    doc_generator.export_to_file("api_spec.json", "json")
    
    # Generate HTML documentation
    html_docs = doc_generator.generate_html_docs()
    with open("api_docs.html", "w") as f:
        f.write(html_docs)
    
    print("Generated API documentation files:")
    print("- api_spec.yaml")
    print("- api_spec.json") 
    print("- api_docs.html")
    
    # Initialize test suite
    test_suite = APITestSuite(
        base_url="http://localhost:8000/api/v1",
        api_key="test-api-key"
    )
    
    # Create sample test cases
    sample_tests = create_sample_test_cases(all_endpoints)
    for test_case in sample_tests:
        test_suite.add_test_case(test_case)
    
    print(f"\nCreated {len(sample_tests)} test cases")
    
    # Run tests (commented out since we don't have a running API server)
    # print("\nRunning API tests...")
    # test_results = await test_suite.run_all_tests()
    # print(f"Test Results: {test_results}")
    
    # Generate test report
    # report = test_suite.generate_test_report()
    # with open("test_report.json", "w") as f:
    #     json.dump(report, f, indent=2)
    
    # Test version management
    version_manager = APIVersionManager()
    version_manager.add_version(APIVersion.V1, doc_generator)
    
    # Create a mock V2 with some changes
    doc_generator_v2 = APIDocumentationGenerator(
        title="Cloud Intelligence Platform API",
        version="2.0.0"
    )
    
    # Add endpoints with some modifications for V2
    v2_endpoints = create_v2_endpoints()
    for endpoint in v2_endpoints:
        doc_generator_v2.add_endpoint(endpoint)
    
    version_manager.add_version(APIVersion.V2, doc_generator_v2)
    
    # Generate compatibility report
    compatibility_report = version_manager.generate_compatibility_report()
    with open("compatibility_report.json", "w") as f:
        json.dump(compatibility_report, f, indent=2)
    
    print("\nGenerated compatibility report: compatibility_report.json")
    
    print("\nAPI documentation and testing tools setup complete!")
    print("\nTo view the interactive API documentation:")
    print("1. Open api_docs.html in a web browser")
    print("2. Or serve the files with a local HTTP server")


def create_sample_test_cases(endpoints):
    """Create sample test cases for endpoints"""
    test_cases = []
    
    for endpoint in endpoints:
        if endpoint.method.value == "GET":
            # Create a basic GET test
            test_case = APITestCase(
                test_id=f"test_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_{endpoint.method.value.lower()}",
                name=f"Test {endpoint.method.value} {endpoint.path}",
                description=f"Test the {endpoint.summary}",
                endpoint=endpoint,
                test_data={},
                expected_status=200
            )
            
            # Add sample data for path parameters
            for param in endpoint.parameters:
                if param.param_type == "path":
                    if "id" in param.name.lower():
                        test_case.test_data[param.name] = "test-123"
                    else:
                        test_case.test_data[param.name] = "sample-value"
                elif param.param_type == "query" and param.required:
                    if param.data_type == "string":
                        if param.format == "date":
                            test_case.test_data[param.name] = "2024-01-01"
                        elif param.format == "date-time":
                            test_case.test_data[param.name] = "2024-01-01T00:00:00Z"
                        else:
                            test_case.test_data[param.name] = "test-value"
                    elif param.data_type == "integer":
                        test_case.test_data[param.name] = 10
                    elif param.data_type == "boolean":
                        test_case.test_data[param.name] = True
            
            test_cases.append(test_case)
            
        elif endpoint.method.value == "POST":
            # Create a basic POST test
            test_case = APITestCase(
                test_id=f"test_{endpoint.path.replace('/', '_')}_{endpoint.method.value.lower()}",
                name=f"Test {endpoint.method.value} {endpoint.path}",
                description=f"Test the {endpoint.summary}",
                endpoint=endpoint,
                test_data={},
                expected_status=201
            )
            
            # Add sample body data
            if "workload" in endpoint.path:
                test_case.test_data = {
                    "workload": {
                        "name": "test-workload",
                        "cpu_request": 2.0,
                        "memory_request": 4.0,
                        "priority": 1
                    }
                }
            elif "budget" in endpoint.path:
                test_case.test_data = {
                    "budget": {
                        "name": "test-budget",
                        "amount": 1000.0,
                        "period": "monthly",
                        "alert_thresholds": [0.8, 0.9]
                    }
                }
            
            test_cases.append(test_case)
    
    return test_cases


def create_v2_endpoints():
    """Create sample V2 endpoints with some changes"""
    from api_documentation import APIEndpoint, HTTPMethod, APIParameter, APIResponse
    
    endpoints = []
    
    # Modified workload endpoint with additional parameter
    endpoints.append(APIEndpoint(
        path="/workloads",
        method=HTTPMethod.GET,
        summary="List all workloads (V2)",
        description="Retrieve a list of all workloads with enhanced filtering",
        tags=["workloads"],
        version=APIVersion.V2,
        parameters=[
            APIParameter("limit", "query", "integer", False, "Maximum number of results", 100),
            APIParameter("offset", "query", "integer", False, "Number of results to skip", 0),
            APIParameter("status", "query", "string", False, "Filter by status", enum_values=["pending", "running", "completed"]),
            APIParameter("provider", "query", "string", False, "Filter by cloud provider")  # New parameter
        ],
        responses=[
            APIResponse(200, "Successful response", example={"workloads": [], "total": 0, "metadata": {}}),
            APIResponse(400, "Bad request"),
            APIResponse(401, "Unauthorized")
        ]
    ))
    
    return endpoints


if __name__ == "__main__":
    asyncio.run(main())