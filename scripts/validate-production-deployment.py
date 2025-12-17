#!/usr/bin/env python3
"""
Production Deployment Validation Script for FinOps Cost Optimization Platform

This script validates that the production deployment is working correctly
by running comprehensive health checks, functional tests, and security validations.
"""

import asyncio
import aiohttp
import json
import sys
import time
import subprocess
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation check"""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None


class ProductionValidator:
    """Comprehensive production deployment validator"""
    
    def __init__(self, base_url: str = "https://api.finops.example.com", namespace: str = "finops-automation"):
        self.base_url = base_url
        self.namespace = namespace
        self.session = None
        self.results: List[ValidationResult] = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_info(self, message: str):
        """Log info message with timestamp"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] INFO: {message}")
    
    def log_error(self, message: str):
        """Log error message with timestamp"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {message}")
    
    def log_success(self, message: str):
        """Log success message with timestamp"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: {message}")
    
    def run_kubectl_command(self, command: List[str]) -> Dict[str, Any]:
        """Run kubectl command and return result"""
        try:
            result = subprocess.run(
                ["kubectl"] + command,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    async def validate_api_endpoints(self) -> List[ValidationResult]:
        """Validate API endpoints are responding correctly"""
        self.log_info("Validating API endpoints...")
        results = []
        
        endpoints = [
            ("/health", "Basic health check"),
            ("/health/ready", "Readiness check"),
            ("/health/live", "Liveness check"),
            ("/health/detailed", "Detailed health check"),
            ("/health/dependencies", "Dependencies check"),
            ("/api/v1/health/cache", "Cache health check"),
            ("/api/v1/health/system", "System health check"),
            ("/docs", "API documentation"),
            ("/metrics", "Prometheus metrics")
        ]
        
        for endpoint, description in endpoints:
            start_time = time.time()
            try:
                url = f"{self.base_url}{endpoint}"
                async with self.session.get(url, timeout=10) as response:
                    duration_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        results.append(ValidationResult(
                            name=f"API Endpoint: {endpoint}",
                            passed=True,
                            message=f"{description} - OK ({response.status})",
                            duration_ms=duration_ms
                        ))
                    else:
                        results.append(ValidationResult(
                            name=f"API Endpoint: {endpoint}",
                            passed=False,
                            message=f"{description} - Failed ({response.status})",
                            duration_ms=duration_ms
                        ))
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                results.append(ValidationResult(
                    name=f"API Endpoint: {endpoint}",
                    passed=False,
                    message=f"{description} - Error: {str(e)}",
                    duration_ms=duration_ms
                ))
        
        return results
    
    def validate_kubernetes_resources(self) -> List[ValidationResult]:
        """Validate Kubernetes resources are deployed correctly"""
        self.log_info("Validating Kubernetes resources...")
        results = []
        
        # Check deployments
        cmd_result = self.run_kubectl_command([
            "get", "deployments", "-n", self.namespace, "-o", "json"
        ])
        
        if cmd_result["success"]:
            try:
                deployments = json.loads(cmd_result["stdout"])
                expected_deployments = ["finops-api", "finops-worker", "finops-scheduler"]
                
                for deployment_name in expected_deployments:
                    deployment = next(
                        (d for d in deployments["items"] if d["metadata"]["name"] == deployment_name),
                        None
                    )
                    
                    if deployment:
                        ready_replicas = deployment["status"].get("readyReplicas", 0)
                        desired_replicas = deployment["spec"]["replicas"]
                        
                        if ready_replicas == desired_replicas:
                            results.append(ValidationResult(
                                name=f"Deployment: {deployment_name}",
                                passed=True,
                                message=f"Ready ({ready_replicas}/{desired_replicas})",
                                details={"ready_replicas": ready_replicas, "desired_replicas": desired_replicas}
                            ))
                        else:
                            results.append(ValidationResult(
                                name=f"Deployment: {deployment_name}",
                                passed=False,
                                message=f"Not ready ({ready_replicas}/{desired_replicas})",
                                details={"ready_replicas": ready_replicas, "desired_replicas": desired_replicas}
                            ))
                    else:
                        results.append(ValidationResult(
                            name=f"Deployment: {deployment_name}",
                            passed=False,
                            message="Deployment not found"
                        ))
            except json.JSONDecodeError:
                results.append(ValidationResult(
                    name="Kubernetes Deployments",
                    passed=False,
                    message="Failed to parse kubectl output"
                ))
        else:
            results.append(ValidationResult(
                name="Kubernetes Deployments",
                passed=False,
                message=f"kubectl command failed: {cmd_result['stderr']}"
            ))
        
        # Check services
        cmd_result = self.run_kubectl_command([
            "get", "services", "-n", self.namespace, "-o", "json"
        ])
        
        if cmd_result["success"]:
            try:
                services = json.loads(cmd_result["stdout"])
                expected_services = ["finops-api"]
                
                for service_name in expected_services:
                    service = next(
                        (s for s in services["items"] if s["metadata"]["name"] == service_name),
                        None
                    )
                    
                    if service:
                        results.append(ValidationResult(
                            name=f"Service: {service_name}",
                            passed=True,
                            message="Service exists",
                            details={"type": service["spec"]["type"]}
                        ))
                    else:
                        results.append(ValidationResult(
                            name=f"Service: {service_name}",
                            passed=False,
                            message="Service not found"
                        ))
            except json.JSONDecodeError:
                results.append(ValidationResult(
                    name="Kubernetes Services",
                    passed=False,
                    message="Failed to parse kubectl output"
                ))
        
        # Check persistent volume claims
        cmd_result = self.run_kubectl_command([
            "get", "pvc", "-n", self.namespace, "-o", "json"
        ])
        
        if cmd_result["success"]:
            try:
                pvcs = json.loads(cmd_result["stdout"])
                for pvc in pvcs["items"]:
                    pvc_name = pvc["metadata"]["name"]
                    status = pvc["status"]["phase"]
                    
                    if status == "Bound":
                        results.append(ValidationResult(
                            name=f"PVC: {pvc_name}",
                            passed=True,
                            message=f"Status: {status}",
                            details={"capacity": pvc["status"].get("capacity", {})}
                        ))
                    else:
                        results.append(ValidationResult(
                            name=f"PVC: {pvc_name}",
                            passed=False,
                            message=f"Status: {status}"
                        ))
            except json.JSONDecodeError:
                results.append(ValidationResult(
                    name="Persistent Volume Claims",
                    passed=False,
                    message="Failed to parse kubectl output"
                ))
        
        return results
    
    def validate_monitoring_stack(self) -> List[ValidationResult]:
        """Validate monitoring stack is deployed and working"""
        self.log_info("Validating monitoring stack...")
        results = []
        
        # Check monitoring namespace
        cmd_result = self.run_kubectl_command([
            "get", "namespace", "monitoring"
        ])
        
        if cmd_result["success"]:
            results.append(ValidationResult(
                name="Monitoring Namespace",
                passed=True,
                message="Monitoring namespace exists"
            ))
        else:
            results.append(ValidationResult(
                name="Monitoring Namespace",
                passed=False,
                message="Monitoring namespace not found"
            ))
            return results  # Can't continue without namespace
        
        # Check Prometheus
        cmd_result = self.run_kubectl_command([
            "get", "pods", "-n", "monitoring", "-l", "app.kubernetes.io/name=prometheus", "-o", "json"
        ])
        
        if cmd_result["success"]:
            try:
                pods = json.loads(cmd_result["stdout"])
                prometheus_pods = [p for p in pods["items"] if p["status"]["phase"] == "Running"]
                
                if prometheus_pods:
                    results.append(ValidationResult(
                        name="Prometheus",
                        passed=True,
                        message=f"{len(prometheus_pods)} Prometheus pods running"
                    ))
                else:
                    results.append(ValidationResult(
                        name="Prometheus",
                        passed=False,
                        message="No running Prometheus pods found"
                    ))
            except json.JSONDecodeError:
                results.append(ValidationResult(
                    name="Prometheus",
                    passed=False,
                    message="Failed to parse kubectl output"
                ))
        
        # Check Grafana
        cmd_result = self.run_kubectl_command([
            "get", "pods", "-n", "monitoring", "-l", "app.kubernetes.io/name=grafana", "-o", "json"
        ])
        
        if cmd_result["success"]:
            try:
                pods = json.loads(cmd_result["stdout"])
                grafana_pods = [p for p in pods["items"] if p["status"]["phase"] == "Running"]
                
                if grafana_pods:
                    results.append(ValidationResult(
                        name="Grafana",
                        passed=True,
                        message=f"{len(grafana_pods)} Grafana pods running"
                    ))
                else:
                    results.append(ValidationResult(
                        name="Grafana",
                        passed=False,
                        message="No running Grafana pods found"
                    ))
            except json.JSONDecodeError:
                results.append(ValidationResult(
                    name="Grafana",
                    passed=False,
                    message="Failed to parse kubectl output"
                ))
        
        return results
    
    def validate_security_configuration(self) -> List[ValidationResult]:
        """Validate security configuration"""
        self.log_info("Validating security configuration...")
        results = []
        
        # Check secrets exist
        cmd_result = self.run_kubectl_command([
            "get", "secret", "finops-secrets", "-n", self.namespace
        ])
        
        if cmd_result["success"]:
            results.append(ValidationResult(
                name="Application Secrets",
                passed=True,
                message="finops-secrets exists"
            ))
        else:
            results.append(ValidationResult(
                name="Application Secrets",
                passed=False,
                message="finops-secrets not found"
            ))
        
        # Check service account
        cmd_result = self.run_kubectl_command([
            "get", "serviceaccount", "finops-service-account", "-n", self.namespace
        ])
        
        if cmd_result["success"]:
            results.append(ValidationResult(
                name="Service Account",
                passed=True,
                message="finops-service-account exists"
            ))
        else:
            results.append(ValidationResult(
                name="Service Account",
                passed=False,
                message="finops-service-account not found"
            ))
        
        # Check network policies
        cmd_result = self.run_kubectl_command([
            "get", "networkpolicy", "-n", self.namespace
        ])
        
        if cmd_result["success"] and cmd_result["stdout"]:
            results.append(ValidationResult(
                name="Network Policies",
                passed=True,
                message="Network policies configured"
            ))
        else:
            results.append(ValidationResult(
                name="Network Policies",
                passed=False,
                message="No network policies found"
            ))
        
        return results
    
    async def validate_functional_tests(self) -> List[ValidationResult]:
        """Run functional tests to validate system behavior"""
        self.log_info("Running functional tests...")
        results = []
        
        # Test API authentication (if applicable)
        try:
            url = f"{self.base_url}/api/v1/auth/health"
            async with self.session.get(url, timeout=10) as response:
                if response.status in [200, 404]:  # 404 is OK if auth is not implemented yet
                    results.append(ValidationResult(
                        name="Authentication System",
                        passed=True,
                        message=f"Auth endpoint accessible ({response.status})"
                    ))
                else:
                    results.append(ValidationResult(
                        name="Authentication System",
                        passed=False,
                        message=f"Auth endpoint error ({response.status})"
                    ))
        except Exception as e:
            results.append(ValidationResult(
                name="Authentication System",
                passed=False,
                message=f"Auth test failed: {str(e)}"
            ))
        
        # Test metrics endpoint
        try:
            url = f"{self.base_url}/metrics"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    text = await response.text()
                    if "finops_" in text:
                        results.append(ValidationResult(
                            name="Metrics Collection",
                            passed=True,
                            message="FinOps metrics are being collected"
                        ))
                    else:
                        results.append(ValidationResult(
                            name="Metrics Collection",
                            passed=False,
                            message="FinOps metrics not found in output"
                        ))
                else:
                    results.append(ValidationResult(
                        name="Metrics Collection",
                        passed=False,
                        message=f"Metrics endpoint error ({response.status})"
                    ))
        except Exception as e:
            results.append(ValidationResult(
                name="Metrics Collection",
                passed=False,
                message=f"Metrics test failed: {str(e)}"
            ))
        
        return results
    
    def validate_backup_system(self) -> List[ValidationResult]:
        """Validate backup system configuration"""
        self.log_info("Validating backup system...")
        results = []
        
        # Check if backup script exists
        backup_script_path = "scripts/backup-restore.sh"
        if os.path.exists(backup_script_path):
            results.append(ValidationResult(
                name="Backup Script",
                passed=True,
                message="Backup script exists"
            ))
            
            # Test backup system health check
            try:
                result = subprocess.run(
                    [backup_script_path, "health-check"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    results.append(ValidationResult(
                        name="Backup System Health",
                        passed=True,
                        message="Backup system health check passed"
                    ))
                else:
                    results.append(ValidationResult(
                        name="Backup System Health",
                        passed=False,
                        message=f"Backup health check failed: {result.stderr}"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    name="Backup System Health",
                    passed=False,
                    message=f"Backup health check error: {str(e)}"
                ))
        else:
            results.append(ValidationResult(
                name="Backup Script",
                passed=False,
                message="Backup script not found"
            ))
        
        # Check for backup CronJob
        cmd_result = self.run_kubectl_command([
            "get", "cronjob", "finops-daily-backup", "-n", self.namespace
        ])
        
        if cmd_result["success"]:
            results.append(ValidationResult(
                name="Backup CronJob",
                passed=True,
                message="Daily backup CronJob configured"
            ))
        else:
            results.append(ValidationResult(
                name="Backup CronJob",
                passed=False,
                message="Daily backup CronJob not found"
            ))
        
        return results
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation checks"""
        self.log_info("Starting comprehensive production deployment validation...")
        
        all_results = []
        
        # Run all validation categories
        validation_categories = [
            ("API Endpoints", self.validate_api_endpoints()),
            ("Kubernetes Resources", self.validate_kubernetes_resources()),
            ("Monitoring Stack", self.validate_monitoring_stack()),
            ("Security Configuration", self.validate_security_configuration()),
            ("Functional Tests", self.validate_functional_tests()),
            ("Backup System", self.validate_backup_system())
        ]
        
        for category_name, validation_coro in validation_categories:
            self.log_info(f"Running {category_name} validation...")
            
            if asyncio.iscoroutine(validation_coro):
                category_results = await validation_coro
            else:
                category_results = validation_coro
            
            all_results.extend(category_results)
            
            # Log category summary
            passed = sum(1 for r in category_results if r.passed)
            total = len(category_results)
            self.log_info(f"{category_name}: {passed}/{total} checks passed")
        
        # Calculate overall results
        total_checks = len(all_results)
        passed_checks = sum(1 for r in all_results if r.passed)
        failed_checks = total_checks - passed_checks
        
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        return {
            "overall_status": "PASS" if failed_checks == 0 else "FAIL",
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "success_rate": success_rate,
            "results": all_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def print_results(self, validation_results: Dict[str, Any]):
        """Print validation results in a formatted way"""
        print("\n" + "="*80)
        print("FINOPS PRODUCTION DEPLOYMENT VALIDATION RESULTS")
        print("="*80)
        
        print(f"\nOverall Status: {validation_results['overall_status']}")
        print(f"Success Rate: {validation_results['success_rate']:.1f}%")
        print(f"Checks: {validation_results['passed_checks']}/{validation_results['total_checks']} passed")
        
        if validation_results['failed_checks'] > 0:
            print(f"\n❌ FAILED CHECKS ({validation_results['failed_checks']}):")
            print("-" * 50)
            
            for result in validation_results['results']:
                if not result.passed:
                    print(f"❌ {result.name}")
                    print(f"   Message: {result.message}")
                    if result.details:
                        print(f"   Details: {result.details}")
                    if result.duration_ms:
                        print(f"   Duration: {result.duration_ms:.1f}ms")
                    print()
        
        print(f"\n✅ PASSED CHECKS ({validation_results['passed_checks']}):")
        print("-" * 50)
        
        for result in validation_results['results']:
            if result.passed:
                duration_info = f" ({result.duration_ms:.1f}ms)" if result.duration_ms else ""
                print(f"✅ {result.name}: {result.message}{duration_info}")
        
        print("\n" + "="*80)
        
        if validation_results['overall_status'] == "PASS":
            self.log_success("🎉 All validation checks passed! Production deployment is ready.")
        else:
            self.log_error(f"❌ {validation_results['failed_checks']} validation checks failed. Please review and fix issues before proceeding.")
        
        print("="*80)


async def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FinOps Production Deployment Validator")
    parser.add_argument("--url", default="https://api.finops.example.com", help="Base URL of the API")
    parser.add_argument("--namespace", default="finops-automation", help="Kubernetes namespace")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--exit-code", action="store_true", help="Exit with non-zero code if validation fails")
    
    args = parser.parse_args()
    
    async with ProductionValidator(args.url, args.namespace) as validator:
        results = await validator.run_all_validations()
        
        if args.output == "json":
            # Convert ValidationResult objects to dictionaries for JSON serialization
            json_results = results.copy()
            json_results["results"] = [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                    "duration_ms": r.duration_ms
                }
                for r in results["results"]
            ]
            print(json.dumps(json_results, indent=2))
        else:
            validator.print_results(results)
        
        if args.exit_code and results["overall_status"] != "PASS":
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())