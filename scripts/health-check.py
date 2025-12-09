#!/usr/bin/env python3
"""
FinOps Platform Health Check Script

This script performs comprehensive health checks on all platform components
and can be used for monitoring, alerting, and troubleshooting.
"""

import asyncio
import aiohttp
import json
import sys
import time
from typing import Dict, Any, List
from datetime import datetime


class HealthChecker:
    """Comprehensive health checker for FinOps platform"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_endpoint(self, endpoint: str, timeout: int = 10) -> Dict[str, Any]:
        """Check a single health endpoint"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            async with self.session.get(url, timeout=timeout) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "healthy",
                        "endpoint": endpoint,
                        "response_time_ms": response_time,
                        "data": data
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "endpoint": endpoint,
                        "response_time_ms": response_time,
                        "http_status": response.status,
                        "error": f"HTTP {response.status}"
                    }
                    
        except asyncio.TimeoutError:
            return {
                "status": "unhealthy",
                "endpoint": endpoint,
                "error": "Timeout",
                "timeout_seconds": timeout
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "endpoint": endpoint,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def check_all_endpoints(self) -> Dict[str, Any]:
        """Check all health endpoints"""
        endpoints = [
            "/health",
            "/health/ready",
            "/health/live", 
            "/health/detailed",
            "/health/dependencies",
            "/api/v1/health/cache",
            "/api/v1/health/system"
        ]
        
        tasks = [self.check_endpoint(endpoint) for endpoint in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        health_results = {}
        overall_status = "healthy"
        
        for i, result in enumerate(results):
            endpoint = endpoints[i]
            
            if isinstance(result, Exception):
                health_results[endpoint] = {
                    "status": "unhealthy",
                    "error": str(result),
                    "error_type": type(result).__name__
                }
                overall_status = "unhealthy"
            else:
                health_results[endpoint] = result
                if result["status"] != "healthy":
                    overall_status = "unhealthy"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": health_results,
            "summary": self._generate_summary(health_results)
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of health check results"""
        total_endpoints = len(results)
        healthy_endpoints = sum(1 for r in results.values() if r["status"] == "healthy")
        
        avg_response_time = 0
        response_times = [r.get("response_time_ms", 0) for r in results.values() if "response_time_ms" in r]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
        
        return {
            "total_endpoints": total_endpoints,
            "healthy_endpoints": healthy_endpoints,
            "unhealthy_endpoints": total_endpoints - healthy_endpoints,
            "health_percentage": (healthy_endpoints / total_endpoints) * 100,
            "average_response_time_ms": avg_response_time
        }


async def main():
    """Main health check function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FinOps Platform Health Checker")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--exit-code", action="store_true", help="Exit with non-zero code if unhealthy")
    
    args = parser.parse_args()
    
    async with HealthChecker(args.url) as checker:
        results = await checker.check_all_endpoints()
        
        if args.format == "json":
            print(json.dumps(results, indent=2))
        else:
            print_text_results(results)
        
        if args.exit_code and results["overall_status"] != "healthy":
            sys.exit(1)


def print_text_results(results: Dict[str, Any]):
    """Print results in human-readable text format"""
    print(f"FinOps Platform Health Check - {results['timestamp']}")
    print(f"Overall Status: {results['overall_status'].upper()}")
    print()
    
    summary = results["summary"]
    print("Summary:")
    print(f"  Total Endpoints: {summary['total_endpoints']}")
    print(f"  Healthy: {summary['healthy_endpoints']}")
    print(f"  Unhealthy: {summary['unhealthy_endpoints']}")
    print(f"  Health Percentage: {summary['health_percentage']:.1f}%")
    print(f"  Average Response Time: {summary['average_response_time_ms']:.1f}ms")
    print()
    
    print("Endpoint Details:")
    for endpoint, result in results["endpoints"].items():
        status_symbol = "✓" if result["status"] == "healthy" else "✗"
        response_time = result.get("response_time_ms", 0)
        
        print(f"  {status_symbol} {endpoint}")
        print(f"    Status: {result['status']}")
        
        if "response_time_ms" in result:
            print(f"    Response Time: {response_time:.1f}ms")
        
        if "error" in result:
            print(f"    Error: {result['error']}")
        
        print()


if __name__ == "__main__":
    asyncio.run(main())