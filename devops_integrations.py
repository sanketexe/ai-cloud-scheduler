# devops_integrations.py
"""
DevOps Tool Integrations

This module provides integration adapters for popular DevOps tools and CI/CD pipelines:
- CI/CD platform integrations (Jenkins, GitLab, GitHub Actions)
- Monitoring tool integrations (Prometheus, Grafana)
- Infrastructure-as-Code integrations (Terraform, CloudFormation)

Requirements addressed:
- 5.4: Integration with CI/CD pipelines
- 5.5: Monitoring tool integrations
"""

import json
import requests
import yaml
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import base64
import os
from urllib.parse import urljoin


class IntegrationType(Enum):
    CICD = "cicd"
    MONITORING = "monitoring"
    IAC = "iac"  # Infrastructure as Code
    NOTIFICATION = "notification"


class IntegrationStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    CONFIGURING = "configuring"


@dataclass
class IntegrationConfig:
    """Base configuration for integrations"""
    integration_id: str
    name: str
    integration_type: IntegrationType
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    credentials: Dict[str, str] = field(default_factory=dict)
    status: IntegrationStatus = IntegrationStatus.INACTIVE
    last_sync: Optional[datetime] = None
    error_message: Optional[str] = None


class BaseIntegration(ABC):
    """Base class for all integrations"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the integration is working"""
        pass
        
    @abstractmethod
    async def sync_data(self) -> Dict[str, Any]:
        """Sync data from the external system"""
        pass
        
    def update_status(self, status: IntegrationStatus, error_message: Optional[str] = None):
        """Update integration status"""
        self.config.status = status
        self.config.error_message = error_message
        self.config.last_sync = datetime.utcnow()


# CI/CD Integrations

class JenkinsIntegration(BaseIntegration):
    """Jenkins CI/CD integration"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.base_url = config.config.get('base_url', '')
        self.username = config.credentials.get('username', '')
        self.api_token = config.credentials.get('api_token', '')
        
    async def test_connection(self) -> bool:
        """Test Jenkins connection"""
        try:
            url = urljoin(self.base_url, '/api/json')
            auth = (self.username, self.api_token)
            
            response = requests.get(url, auth=auth, timeout=10)
            if response.status_code == 200:
                self.update_status(IntegrationStatus.ACTIVE)
                return True
            else:
                self.update_status(IntegrationStatus.ERROR, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return False
            
    async def sync_data(self) -> Dict[str, Any]:
        """Sync Jenkins job data"""
        try:
            # Get all jobs
            jobs_url = urljoin(self.base_url, '/api/json?tree=jobs[name,url,lastBuild[number,result,timestamp]]')
            auth = (self.username, self.api_token)
            
            response = requests.get(jobs_url, auth=auth, timeout=30)
            response.raise_for_status()
            
            jobs_data = response.json()
            
            # Process job information
            processed_jobs = []
            for job in jobs_data.get('jobs', []):
                last_build = job.get('lastBuild')
                processed_job = {
                    'name': job['name'],
                    'url': job['url'],
                    'last_build': {
                        'number': last_build.get('number') if last_build else None,
                        'result': last_build.get('result') if last_build else None,
                        'timestamp': last_build.get('timestamp') if last_build else None
                    } if last_build else None
                }
                processed_jobs.append(processed_job)
                
            self.update_status(IntegrationStatus.ACTIVE)
            return {'jobs': processed_jobs, 'total_jobs': len(processed_jobs)}
            
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return {'error': str(e)}
            
    async def trigger_build(self, job_name: str, parameters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Trigger a Jenkins build"""
        try:
            if parameters:
                url = urljoin(self.base_url, f'/job/{job_name}/buildWithParameters')
                data = parameters
            else:
                url = urljoin(self.base_url, f'/job/{job_name}/build')
                data = {}
                
            auth = (self.username, self.api_token)
            response = requests.post(url, data=data, auth=auth, timeout=30)
            
            if response.status_code in [200, 201]:
                return {'success': True, 'message': 'Build triggered successfully'}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}


class GitLabIntegration(BaseIntegration):
    """GitLab CI/CD integration"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.base_url = config.config.get('base_url', 'https://gitlab.com')
        self.access_token = config.credentials.get('access_token', '')
        self.project_id = config.config.get('project_id', '')
        
    async def test_connection(self) -> bool:
        """Test GitLab connection"""
        try:
            url = urljoin(self.base_url, f'/api/v4/projects/{self.project_id}')
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                self.update_status(IntegrationStatus.ACTIVE)
                return True
            else:
                self.update_status(IntegrationStatus.ERROR, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return False
            
    async def sync_data(self) -> Dict[str, Any]:
        """Sync GitLab pipeline data"""
        try:
            url = urljoin(self.base_url, f'/api/v4/projects/{self.project_id}/pipelines')
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            pipelines = response.json()
            
            # Process pipeline information
            processed_pipelines = []
            for pipeline in pipelines[:10]:  # Get last 10 pipelines
                processed_pipeline = {
                    'id': pipeline['id'],
                    'status': pipeline['status'],
                    'ref': pipeline['ref'],
                    'created_at': pipeline['created_at'],
                    'updated_at': pipeline['updated_at'],
                    'web_url': pipeline['web_url']
                }
                processed_pipelines.append(processed_pipeline)
                
            self.update_status(IntegrationStatus.ACTIVE)
            return {'pipelines': processed_pipelines, 'total_pipelines': len(processed_pipelines)}
            
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return {'error': str(e)}
            
    async def trigger_pipeline(self, ref: str = 'main', variables: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Trigger a GitLab pipeline"""
        try:
            url = urljoin(self.base_url, f'/api/v4/projects/{self.project_id}/pipeline')
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            data = {'ref': ref}
            if variables:
                data['variables'] = [{'key': k, 'value': v} for k, v in variables.items()]
                
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 201:
                pipeline_data = response.json()
                return {
                    'success': True, 
                    'pipeline_id': pipeline_data['id'],
                    'web_url': pipeline_data['web_url']
                }
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}


class GitHubActionsIntegration(BaseIntegration):
    """GitHub Actions integration"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.base_url = 'https://api.github.com'
        self.access_token = config.credentials.get('access_token', '')
        self.owner = config.config.get('owner', '')
        self.repo = config.config.get('repo', '')
        
    async def test_connection(self) -> bool:
        """Test GitHub connection"""
        try:
            url = f'{self.base_url}/repos/{self.owner}/{self.repo}'
            headers = {'Authorization': f'token {self.access_token}'}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                self.update_status(IntegrationStatus.ACTIVE)
                return True
            else:
                self.update_status(IntegrationStatus.ERROR, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return False
            
    async def sync_data(self) -> Dict[str, Any]:
        """Sync GitHub Actions workflow runs"""
        try:
            url = f'{self.base_url}/repos/{self.owner}/{self.repo}/actions/runs'
            headers = {'Authorization': f'token {self.access_token}'}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            runs_data = response.json()
            
            # Process workflow runs
            processed_runs = []
            for run in runs_data.get('workflow_runs', [])[:10]:  # Get last 10 runs
                processed_run = {
                    'id': run['id'],
                    'name': run['name'],
                    'status': run['status'],
                    'conclusion': run['conclusion'],
                    'created_at': run['created_at'],
                    'updated_at': run['updated_at'],
                    'html_url': run['html_url'],
                    'head_branch': run['head_branch']
                }
                processed_runs.append(processed_run)
                
            self.update_status(IntegrationStatus.ACTIVE)
            return {'workflow_runs': processed_runs, 'total_runs': len(processed_runs)}
            
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return {'error': str(e)}
            
    async def trigger_workflow(self, workflow_id: str, ref: str = 'main', inputs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Trigger a GitHub Actions workflow"""
        try:
            url = f'{self.base_url}/repos/{self.owner}/{self.repo}/actions/workflows/{workflow_id}/dispatches'
            headers = {'Authorization': f'token {self.access_token}'}
            
            data = {'ref': ref}
            if inputs:
                data['inputs'] = inputs
                
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 204:
                return {'success': True, 'message': 'Workflow triggered successfully'}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Monitoring Integrations

class PrometheusIntegration(BaseIntegration):
    """Prometheus monitoring integration"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.base_url = config.config.get('base_url', '')
        self.username = config.credentials.get('username', '')
        self.password = config.credentials.get('password', '')
        
    async def test_connection(self) -> bool:
        """Test Prometheus connection"""
        try:
            url = urljoin(self.base_url, '/api/v1/query?query=up')
            auth = (self.username, self.password) if self.username else None
            
            response = requests.get(url, auth=auth, timeout=10)
            if response.status_code == 200:
                self.update_status(IntegrationStatus.ACTIVE)
                return True
            else:
                self.update_status(IntegrationStatus.ERROR, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return False
            
    async def sync_data(self) -> Dict[str, Any]:
        """Sync Prometheus metrics"""
        try:
            # Query for basic system metrics
            queries = {
                'cpu_usage': 'avg(100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))',
                'memory_usage': 'avg((1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100)',
                'disk_usage': 'avg((1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100)',
                'up_instances': 'count(up == 1)'
            }
            
            auth = (self.username, self.password) if self.username else None
            metrics = {}
            
            for metric_name, query in queries.items():
                url = urljoin(self.base_url, f'/api/v1/query?query={query}')
                response = requests.get(url, auth=auth, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success' and data['data']['result']:
                        metrics[metric_name] = float(data['data']['result'][0]['value'][1])
                    else:
                        metrics[metric_name] = None
                else:
                    metrics[metric_name] = None
                    
            self.update_status(IntegrationStatus.ACTIVE)
            return {'metrics': metrics, 'timestamp': datetime.utcnow().isoformat()}
            
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return {'error': str(e)}
            
    async def query_metric(self, query: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        """Execute a custom Prometheus query"""
        try:
            if start_time and end_time:
                url = urljoin(self.base_url, f'/api/v1/query_range?query={query}&start={start_time}&end={end_time}&step=60s')
            else:
                url = urljoin(self.base_url, f'/api/v1/query?query={query}')
                
            auth = (self.username, self.password) if self.username else None
            response = requests.get(url, auth=auth, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}


class GrafanaIntegration(BaseIntegration):
    """Grafana integration for dashboard management"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.base_url = config.config.get('base_url', '')
        self.api_key = config.credentials.get('api_key', '')
        
    async def test_connection(self) -> bool:
        """Test Grafana connection"""
        try:
            url = urljoin(self.base_url, '/api/org')
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                self.update_status(IntegrationStatus.ACTIVE)
                return True
            else:
                self.update_status(IntegrationStatus.ERROR, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return False
            
    async def sync_data(self) -> Dict[str, Any]:
        """Sync Grafana dashboards"""
        try:
            url = urljoin(self.base_url, '/api/search?type=dash-db')
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            dashboards = response.json()
            
            # Process dashboard information
            processed_dashboards = []
            for dashboard in dashboards:
                processed_dashboard = {
                    'id': dashboard['id'],
                    'uid': dashboard['uid'],
                    'title': dashboard['title'],
                    'uri': dashboard['uri'],
                    'url': dashboard['url'],
                    'type': dashboard['type'],
                    'tags': dashboard.get('tags', [])
                }
                processed_dashboards.append(processed_dashboard)
                
            self.update_status(IntegrationStatus.ACTIVE)
            return {'dashboards': processed_dashboards, 'total_dashboards': len(processed_dashboards)}
            
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return {'error': str(e)}
            
    async def create_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Grafana dashboard"""
        try:
            url = urljoin(self.base_url, '/api/dashboards/db')
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, json=dashboard_config, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'id': result['id'],
                    'uid': result['uid'],
                    'url': result['url']
                }
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Infrastructure as Code Integrations

class TerraformIntegration(BaseIntegration):
    """Terraform integration for infrastructure management"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.workspace_path = config.config.get('workspace_path', '')
        self.terraform_binary = config.config.get('terraform_binary', 'terraform')
        
    async def test_connection(self) -> bool:
        """Test Terraform installation and workspace"""
        try:
            import subprocess
            
            # Check if terraform is installed
            result = subprocess.run([self.terraform_binary, 'version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self.update_status(IntegrationStatus.ACTIVE)
                return True
            else:
                self.update_status(IntegrationStatus.ERROR, "Terraform not found or not working")
                return False
                
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return False
            
    async def sync_data(self) -> Dict[str, Any]:
        """Get Terraform state information"""
        try:
            import subprocess
            
            # Get terraform state
            result = subprocess.run([self.terraform_binary, 'show', '-json'], 
                                  cwd=self.workspace_path,
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                state_data = json.loads(result.stdout)
                
                # Extract resource information
                resources = []
                if 'values' in state_data and 'root_module' in state_data['values']:
                    root_module = state_data['values']['root_module']
                    if 'resources' in root_module:
                        for resource in root_module['resources']:
                            resources.append({
                                'address': resource['address'],
                                'type': resource['type'],
                                'name': resource['name'],
                                'provider_name': resource['provider_name']
                            })
                            
                self.update_status(IntegrationStatus.ACTIVE)
                return {
                    'resources': resources,
                    'total_resources': len(resources),
                    'terraform_version': state_data.get('terraform_version', 'unknown')
                }
            else:
                self.update_status(IntegrationStatus.ERROR, result.stderr)
                return {'error': result.stderr}
                
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return {'error': str(e)}
            
    async def plan_infrastructure(self) -> Dict[str, Any]:
        """Run terraform plan"""
        try:
            import subprocess
            
            result = subprocess.run([self.terraform_binary, 'plan', '-json'], 
                                  cwd=self.workspace_path,
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return {'success': True, 'output': result.stdout}
            else:
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def apply_infrastructure(self, auto_approve: bool = False) -> Dict[str, Any]:
        """Run terraform apply"""
        try:
            import subprocess
            
            cmd = [self.terraform_binary, 'apply']
            if auto_approve:
                cmd.append('-auto-approve')
                
            result = subprocess.run(cmd, 
                                  cwd=self.workspace_path,
                                  capture_output=True, text=True, timeout=1800)  # 30 minutes
            
            if result.returncode == 0:
                return {'success': True, 'output': result.stdout}
            else:
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}


class CloudFormationIntegration(BaseIntegration):
    """AWS CloudFormation integration"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.region = config.config.get('region', 'us-east-1')
        self.access_key = config.credentials.get('access_key', '')
        self.secret_key = config.credentials.get('secret_key', '')
        
    async def test_connection(self) -> bool:
        """Test AWS CloudFormation connection"""
        try:
            import boto3
            
            session = boto3.Session(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            
            cf_client = session.client('cloudformation')
            cf_client.list_stacks(MaxItems=1)
            
            self.update_status(IntegrationStatus.ACTIVE)
            return True
            
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return False
            
    async def sync_data(self) -> Dict[str, Any]:
        """Sync CloudFormation stacks"""
        try:
            import boto3
            
            session = boto3.Session(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            
            cf_client = session.client('cloudformation')
            response = cf_client.list_stacks(
                StackStatusFilter=[
                    'CREATE_COMPLETE', 'UPDATE_COMPLETE', 'DELETE_FAILED',
                    'CREATE_FAILED', 'UPDATE_FAILED', 'ROLLBACK_COMPLETE'
                ]
            )
            
            stacks = []
            for stack in response['StackSummaries']:
                stacks.append({
                    'stack_name': stack['StackName'],
                    'stack_status': stack['StackStatus'],
                    'creation_time': stack['CreationTime'].isoformat(),
                    'last_updated_time': stack.get('LastUpdatedTime', '').isoformat() if stack.get('LastUpdatedTime') else None,
                    'template_description': stack.get('TemplateDescription', '')
                })
                
            self.update_status(IntegrationStatus.ACTIVE)
            return {'stacks': stacks, 'total_stacks': len(stacks)}
            
        except Exception as e:
            self.update_status(IntegrationStatus.ERROR, str(e))
            return {'error': str(e)}


# Integration Manager

class DevOpsIntegrationManager:
    """Manages all DevOps tool integrations"""
    
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.logger = logging.getLogger(__name__)
        
        # Integration type mapping
        self.integration_classes = {
            'jenkins': JenkinsIntegration,
            'gitlab': GitLabIntegration,
            'github_actions': GitHubActionsIntegration,
            'prometheus': PrometheusIntegration,
            'grafana': GrafanaIntegration,
            'terraform': TerraformIntegration,
            'cloudformation': CloudFormationIntegration
        }
        
    def add_integration(self, integration_type: str, config: IntegrationConfig) -> bool:
        """Add a new integration"""
        try:
            if integration_type not in self.integration_classes:
                self.logger.error(f"Unknown integration type: {integration_type}")
                return False
                
            integration_class = self.integration_classes[integration_type]
            integration = integration_class(config)
            
            self.integrations[config.integration_id] = integration
            self.logger.info(f"Added {integration_type} integration: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add integration {config.name}: {e}")
            return False
            
    def remove_integration(self, integration_id: str) -> bool:
        """Remove an integration"""
        if integration_id in self.integrations:
            integration = self.integrations.pop(integration_id)
            self.logger.info(f"Removed integration: {integration.config.name}")
            return True
        return False
        
    async def test_all_integrations(self) -> Dict[str, bool]:
        """Test all configured integrations"""
        results = {}
        for integration_id, integration in self.integrations.items():
            try:
                results[integration_id] = await integration.test_connection()
            except Exception as e:
                self.logger.error(f"Error testing integration {integration_id}: {e}")
                results[integration_id] = False
        return results
        
    async def sync_all_data(self) -> Dict[str, Any]:
        """Sync data from all active integrations"""
        results = {}
        for integration_id, integration in self.integrations.items():
            if integration.config.enabled and integration.config.status == IntegrationStatus.ACTIVE:
                try:
                    results[integration_id] = await integration.sync_data()
                except Exception as e:
                    self.logger.error(f"Error syncing data from {integration_id}: {e}")
                    results[integration_id] = {'error': str(e)}
        return results
        
    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all integrations"""
        status = {}
        for integration_id, integration in self.integrations.items():
            status[integration_id] = {
                'name': integration.config.name,
                'type': integration.config.integration_type.value,
                'status': integration.config.status.value,
                'enabled': integration.config.enabled,
                'last_sync': integration.config.last_sync.isoformat() if integration.config.last_sync else None,
                'error_message': integration.config.error_message
            }
        return status
        
    def get_integration(self, integration_id: str) -> Optional[BaseIntegration]:
        """Get a specific integration"""
        return self.integrations.get(integration_id)


# Convenience functions for creating integration configs

def create_jenkins_config(name: str, base_url: str, username: str, api_token: str) -> IntegrationConfig:
    """Create Jenkins integration configuration"""
    return IntegrationConfig(
        integration_id=f"jenkins_{name.lower().replace(' ', '_')}",
        name=name,
        integration_type=IntegrationType.CICD,
        config={'base_url': base_url},
        credentials={'username': username, 'api_token': api_token}
    )


def create_gitlab_config(name: str, base_url: str, access_token: str, project_id: str) -> IntegrationConfig:
    """Create GitLab integration configuration"""
    return IntegrationConfig(
        integration_id=f"gitlab_{name.lower().replace(' ', '_')}",
        name=name,
        integration_type=IntegrationType.CICD,
        config={'base_url': base_url, 'project_id': project_id},
        credentials={'access_token': access_token}
    )


def create_github_config(name: str, access_token: str, owner: str, repo: str) -> IntegrationConfig:
    """Create GitHub Actions integration configuration"""
    return IntegrationConfig(
        integration_id=f"github_{name.lower().replace(' ', '_')}",
        name=name,
        integration_type=IntegrationType.CICD,
        config={'owner': owner, 'repo': repo},
        credentials={'access_token': access_token}
    )


def create_prometheus_config(name: str, base_url: str, username: str = '', password: str = '') -> IntegrationConfig:
    """Create Prometheus integration configuration"""
    return IntegrationConfig(
        integration_id=f"prometheus_{name.lower().replace(' ', '_')}",
        name=name,
        integration_type=IntegrationType.MONITORING,
        config={'base_url': base_url},
        credentials={'username': username, 'password': password}
    )


def create_grafana_config(name: str, base_url: str, api_key: str) -> IntegrationConfig:
    """Create Grafana integration configuration"""
    return IntegrationConfig(
        integration_id=f"grafana_{name.lower().replace(' ', '_')}",
        name=name,
        integration_type=IntegrationType.MONITORING,
        config={'base_url': base_url},
        credentials={'api_key': api_key}
    )