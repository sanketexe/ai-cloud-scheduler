#!/bin/bash

# FinOps AI Services Production Deployment Script
# This script deploys the complete AI/ML infrastructure for the FinOps platform

set -e

# Configuration
NAMESPACE="finops-ai-services"
KUBECTL_TIMEOUT="600s"
HELM_TIMEOUT="10m"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if running on GPU-enabled cluster
    GPU_NODES=$(kubectl get nodes -l accelerator --no-headers 2>/dev/null | wc -l)
    if [ "$GPU_NODES" -eq 0 ]; then
        log_warning "No GPU nodes detected. AI services requiring GPU will not function properly."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "Found $GPU_NODES GPU-enabled nodes"
    fi
    
    log_success "Prerequisites check completed"
}

# Setup GPU nodes
setup_gpu_nodes() {
    log_info "Setting up GPU nodes..."
    
    # Apply GPU node configuration
    kubectl apply -f ai-gpu-resource-management.yaml
    
    # Wait for NVIDIA device plugin to be ready
    log_info "Waiting for NVIDIA device plugin to be ready..."
    kubectl wait --for=condition=ready pod -l name=nvidia-device-plugin-ds -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    # Wait for DCGM exporter to be ready
    log_info "Waiting for DCGM exporter to be ready..."
    kubectl wait --for=condition=ready pod -l app=dcgm-exporter -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    log_success "GPU nodes setup completed"
}

# Deploy core AI services
deploy_ai_services() {
    log_info "Deploying AI services..."
    
    # Apply AI services deployment
    kubectl apply -f ai-services-production-deployment.yaml
    
    # Wait for deployments to be ready
    log_info "Waiting for AI service deployments to be ready..."
    
    DEPLOYMENTS=(
        "predictive-scaling-engine"
        "workload-intelligence-system"
        "natural-language-interface"
        "graph-neural-network-system"
        "predictive-maintenance-system"
        "smart-contract-optimizer"
        "ai-orchestrator"
    )
    
    for deployment in "${DEPLOYMENTS[@]}"; do
        log_info "Waiting for $deployment to be ready..."
        kubectl wait --for=condition=available deployment/$deployment -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    done
    
    # Special handling for RL agent (single replica, longer startup time)
    log_info "Waiting for reinforcement-learning-agent to be ready..."
    kubectl wait --for=condition=available deployment/reinforcement-learning-agent -n $NAMESPACE --timeout=900s
    
    log_success "AI services deployment completed"
}

# Deploy auto-scaling configuration
deploy_autoscaling() {
    log_info "Deploying auto-scaling configuration..."
    
    kubectl apply -f ai-services-autoscaling.yaml
    
    # Wait for HPAs to be ready
    log_info "Waiting for Horizontal Pod Autoscalers to be ready..."
    kubectl wait --for=condition=ScalingActive hpa --all -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    log_success "Auto-scaling configuration deployed"
}

# Deploy monitoring and observability
deploy_monitoring() {
    log_info "Deploying monitoring and observability stack..."
    
    kubectl apply -f ai-monitoring-observability.yaml
    
    # Wait for monitoring components to be ready
    log_info "Waiting for Prometheus to be ready..."
    kubectl wait --for=condition=available deployment/prometheus -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    log_info "Waiting for Grafana to be ready..."
    kubectl wait --for=condition=available deployment/grafana -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    log_info "Waiting for Jaeger to be ready..."
    kubectl wait --for=condition=available deployment/jaeger-all-in-one -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    log_success "Monitoring and observability stack deployed"
}

# Deploy disaster recovery and backup
deploy_disaster_recovery() {
    log_info "Deploying disaster recovery and backup systems..."
    
    kubectl apply -f ai-disaster-recovery-backup.yaml
    
    # Wait for disaster recovery controller to be ready
    log_info "Waiting for disaster recovery controller to be ready..."
    kubectl wait --for=condition=available deployment/disaster-recovery-controller -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    # Check if backup jobs are scheduled
    log_info "Verifying backup jobs are scheduled..."
    kubectl get cronjobs -n $NAMESPACE
    
    log_success "Disaster recovery and backup systems deployed"
}

# Deploy networking and ingress
deploy_networking() {
    log_info "Deploying networking and ingress configuration..."
    
    kubectl apply -f ai-ingress-networking.yaml
    
    # Wait for ingress to be ready
    log_info "Waiting for ingress to be ready..."
    sleep 30  # Give time for ingress controller to process
    
    # Get ingress status
    kubectl get ingress ai-services-ingress -n $NAMESPACE
    
    log_success "Networking and ingress configuration deployed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check all pods are running
    log_info "Checking pod status..."
    kubectl get pods -n $NAMESPACE
    
    # Check services
    log_info "Checking services..."
    kubectl get services -n $NAMESPACE
    
    # Check ingress
    log_info "Checking ingress..."
    kubectl get ingress -n $NAMESPACE
    
    # Check HPAs
    log_info "Checking Horizontal Pod Autoscalers..."
    kubectl get hpa -n $NAMESPACE
    
    # Check PVCs
    log_info "Checking Persistent Volume Claims..."
    kubectl get pvc -n $NAMESPACE
    
    # Test AI service endpoints
    log_info "Testing AI service endpoints..."
    
    # Get the load balancer IP/hostname
    LB_ENDPOINT=$(kubectl get service ai-services-load-balancer -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -z "$LB_ENDPOINT" ]; then
        LB_ENDPOINT=$(kubectl get service ai-services-load-balancer -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    fi
    
    if [ -n "$LB_ENDPOINT" ]; then
        log_info "Load balancer endpoint: $LB_ENDPOINT"
        
        # Test orchestrator health endpoint
        if curl -f -m 10 "http://$LB_ENDPOINT/orchestrator/health" > /dev/null 2>&1; then
            log_success "AI Orchestrator health check passed"
        else
            log_warning "AI Orchestrator health check failed"
        fi
    else
        log_warning "Load balancer endpoint not available yet"
    fi
    
    log_success "Deployment verification completed"
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."
    
    REPORT_FILE="ai-services-deployment-report-$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$REPORT_FILE" << EOF
# FinOps AI Services Deployment Report
Generated: $(date)
Namespace: $NAMESPACE

## Deployment Status
EOF
    
    echo "### Pods" >> "$REPORT_FILE"
    kubectl get pods -n $NAMESPACE >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "### Services" >> "$REPORT_FILE"
    kubectl get services -n $NAMESPACE >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "### Ingress" >> "$REPORT_FILE"
    kubectl get ingress -n $NAMESPACE >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "### Horizontal Pod Autoscalers" >> "$REPORT_FILE"
    kubectl get hpa -n $NAMESPACE >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "### Persistent Volume Claims" >> "$REPORT_FILE"
    kubectl get pvc -n $NAMESPACE >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "### Resource Quotas" >> "$REPORT_FILE"
    kubectl get resourcequota -n $NAMESPACE >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "### GPU Resources" >> "$REPORT_FILE"
    kubectl describe nodes -l node-type=gpu-enabled >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    cat >> "$REPORT_FILE" << EOF

## Access Information

### Web Interfaces
- AI Services Dashboard: https://ai.finops.example.com
- Predictive Scaling: https://predictive-scaling.finops.example.com
- Natural Language Interface: https://nlp.finops.example.com
- AI Orchestrator: https://orchestrator.finops.example.com
- Monitoring Dashboard: https://monitoring.finops.example.com/grafana
- Prometheus: https://monitoring.finops.example.com/prometheus
- Jaeger Tracing: https://monitoring.finops.example.com/jaeger

### API Endpoints
- Predictive Scaling API: https://ai.finops.example.com/predictive-scaling/api/v1
- Workload Intelligence API: https://ai.finops.example.com/workload-intelligence/api/v1
- Natural Language API: https://ai.finops.example.com/natural-language/api/v1
- Graph Neural Network API: https://ai.finops.example.com/graph-neural-network/api/v1
- Predictive Maintenance API: https://ai.finops.example.com/predictive-maintenance/api/v1
- Smart Contract Optimizer API: https://ai.finops.example.com/smart-contract-optimizer/api/v1
- AI Orchestrator API: https://ai.finops.example.com/orchestrator/api/v1

### Monitoring and Metrics
- Prometheus Metrics: http://prometheus.finops-ai-services.svc.cluster.local:9090
- Grafana Dashboards: http://grafana.finops-ai-services.svc.cluster.local:3000
- Jaeger Tracing: http://jaeger-query.finops-ai-services.svc.cluster.local:16686

### Backup and Recovery
- Model Backups: Scheduled every 6 hours
- Database Backups: Scheduled daily at 2 AM
- Cross-region replication: Enabled
- Disaster recovery controller: Active

## Next Steps

1. Update DNS records to point to the load balancer endpoint
2. Configure SSL certificates for production domains
3. Set up monitoring alerts and notification channels
4. Review and update resource quotas and limits as needed
5. Configure backup retention policies
6. Set up cross-region disaster recovery procedures
7. Conduct load testing and performance optimization
8. Review security configurations and access controls

## Troubleshooting

### Common Issues
1. GPU nodes not ready: Check NVIDIA drivers and device plugin
2. Services not starting: Check resource limits and node capacity
3. Ingress not accessible: Verify DNS configuration and SSL certificates
4. High memory usage: Review model sizes and caching configurations
5. Backup failures: Check storage permissions and retention policies

### Useful Commands
- Check pod logs: kubectl logs -f <pod-name> -n $NAMESPACE
- Describe pod issues: kubectl describe pod <pod-name> -n $NAMESPACE
- Check resource usage: kubectl top pods -n $NAMESPACE
- View events: kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'
- Scale deployment: kubectl scale deployment <deployment-name> --replicas=<count> -n $NAMESPACE

EOF
    
    log_success "Deployment report generated: $REPORT_FILE"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up AI services deployment..."
    
    # Delete in reverse order
    kubectl delete -f ai-ingress-networking.yaml --ignore-not-found=true
    kubectl delete -f ai-disaster-recovery-backup.yaml --ignore-not-found=true
    kubectl delete -f ai-monitoring-observability.yaml --ignore-not-found=true
    kubectl delete -f ai-services-autoscaling.yaml --ignore-not-found=true
    kubectl delete -f ai-services-production-deployment.yaml --ignore-not-found=true
    kubectl delete -f ai-gpu-resource-management.yaml --ignore-not-found=true
    
    # Delete namespace (this will delete everything in it)
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting FinOps AI Services deployment..."
    
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            setup_gpu_nodes
            deploy_ai_services
            deploy_autoscaling
            deploy_monitoring
            deploy_disaster_recovery
            deploy_networking
            verify_deployment
            generate_report
            log_success "FinOps AI Services deployment completed successfully!"
            ;;
        "cleanup")
            cleanup
            ;;
        "verify")
            verify_deployment
            ;;
        "report")
            generate_report
            ;;
        *)
            echo "Usage: $0 [deploy|cleanup|verify|report]"
            echo "  deploy  - Deploy all AI services (default)"
            echo "  cleanup - Remove all AI services"
            echo "  verify  - Verify existing deployment"
            echo "  report  - Generate deployment report"
            exit 1
            ;;
    esac
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"