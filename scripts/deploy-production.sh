#!/bin/bash
# Production Deployment Script for FinOps Automated Cost Optimization Platform
# This script handles production deployment with proper IAM role setup and monitoring

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-finops-production}"
NAMESPACE="${NAMESPACE:-finops-automation}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${1}"
}

log_info() {
    log "${BLUE}INFO${NC}: $1"
}

log_success() {
    log "${GREEN}SUCCESS${NC}: $1"
}

log_warning() {
    log "${YELLOW}WARNING${NC}: $1"
}

log_error() {
    log "${RED}ERROR${NC}: $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "kubectl" "aws" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "$tool is required but not installed"
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error_exit "AWS credentials not configured or invalid"
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running"
    fi
    
    log_success "Prerequisites check passed"
}

# Setup IAM roles and policies for automated cost optimization
setup_iam_roles() {
    log_info "Setting up IAM roles and policies..."
    
    # Create IAM policy for cost optimization actions
    cat > /tmp/cost-optimization-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceStatus",
                "ec2:DescribeVolumes",
                "ec2:DescribeSnapshots",
                "ec2:DescribeAddresses",
                "ec2:DescribeLoadBalancers",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeNetworkInterfaces",
                "ec2:StopInstances",
                "ec2:StartInstances",
                "ec2:ModifyInstanceAttribute",
                "ec2:CreateSnapshot",
                "ec2:DeleteVolume",
                "ec2:ModifyVolume",
                "ec2:ReleaseAddress",
                "ec2:DeleteLoadBalancer",
                "ec2:DeleteSecurityGroup",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:GetMetricData",
                "ce:GetCostAndUsage",
                "ce:GetUsageReport",
                "ce:GetReservationCoverage",
                "ce:GetReservationPurchaseRecommendation",
                "ce:GetReservationUtilization",
                "pricing:GetProducts",
                "pricing:GetAttributeValues",
                "support:DescribeTrustedAdvisorChecks",
                "support:DescribeTrustedAdvisorCheckResult",
                "sts:AssumeRole"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::*:role/FinOps-*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogGroups",
                "logs:DescribeLogStreams"
            ],
            "Resource": "arn:aws:logs:*:*:log-group:/finops/*"
        }
    ]
}
EOF

    # Create or update IAM policy
    local policy_arn="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/FinOpsCostOptimizationPolicy"
    
    if aws iam get-policy --policy-arn "$policy_arn" &> /dev/null; then
        log_info "Updating existing IAM policy..."
        aws iam create-policy-version \
            --policy-arn "$policy_arn" \
            --policy-document file:///tmp/cost-optimization-policy.json \
            --set-as-default
    else
        log_info "Creating new IAM policy..."
        aws iam create-policy \
            --policy-name "FinOpsCostOptimizationPolicy" \
            --policy-document file:///tmp/cost-optimization-policy.json \
            --description "Policy for FinOps automated cost optimization actions"
    fi
    
    # Create service role for EKS
    cat > /tmp/trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "eks.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::ACCOUNT_ID:root"
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "sts:ExternalId": "finops-cost-optimization"
                }
            }
        }
    ]
}
EOF

    # Replace ACCOUNT_ID placeholder
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    sed -i "s/ACCOUNT_ID/$account_id/g" /tmp/trust-policy.json
    
    # Create or update service role
    local role_name="FinOpsCostOptimizationRole"
    if aws iam get-role --role-name "$role_name" &> /dev/null; then
        log_info "Updating existing service role..."
        aws iam update-assume-role-policy \
            --role-name "$role_name" \
            --policy-document file:///tmp/trust-policy.json
    else
        log_info "Creating new service role..."
        aws iam create-role \
            --role-name "$role_name" \
            --assume-role-policy-document file:///tmp/trust-policy.json \
            --description "Service role for FinOps cost optimization automation"
    fi
    
    # Attach policy to role
    aws iam attach-role-policy \
        --role-name "$role_name" \
        --policy-arn "$policy_arn"
    
    # Clean up temporary files
    rm -f /tmp/cost-optimization-policy.json /tmp/trust-policy.json
    
    log_success "IAM roles and policies configured"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    local registry="${ECR_REGISTRY:-$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com}"
    local image_tag="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$registry"
    
    # Create ECR repositories if they don't exist
    local repositories=("finops/api" "finops/frontend" "finops/worker")
    for repo in "${repositories[@]}"; do
        if ! aws ecr describe-repositories --repository-names "$repo" --region "$AWS_REGION" &> /dev/null; then
            log_info "Creating ECR repository: $repo"
            aws ecr create-repository --repository-name "$repo" --region "$AWS_REGION"
        fi
    done
    
    # Build and push API image
    log_info "Building API image..."
    docker build -t "$registry/finops/api:$image_tag" -f Dockerfile .
    docker push "$registry/finops/api:$image_tag"
    
    # Build and push frontend image
    log_info "Building frontend image..."
    docker build -t "$registry/finops/frontend:$image_tag" -f frontend/Dockerfile ./frontend
    docker push "$registry/finops/frontend:$image_tag"
    
    # Tag as latest
    docker tag "$registry/finops/api:$image_tag" "$registry/finops/api:latest"
    docker tag "$registry/finops/frontend:$image_tag" "$registry/finops/frontend:latest"
    docker push "$registry/finops/api:latest"
    docker push "$registry/finops/frontend:latest"
    
    log_success "Docker images built and pushed"
    echo "API_IMAGE=$registry/finops/api:$image_tag" > /tmp/deployment-vars.env
    echo "FRONTEND_IMAGE=$registry/finops/frontend:$image_tag" >> /tmp/deployment-vars.env
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Update kubeconfig
    aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    
    # Load deployment variables
    if [[ -f /tmp/deployment-vars.env ]]; then
        source /tmp/deployment-vars.env
    fi
    
    # Apply manifests with environment substitution
    envsubst < "$PROJECT_ROOT/k8s/finops-production-deployment.yaml" | kubectl apply -f -
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/finops-api -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=600s deployment/finops-frontend -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=600s deployment/finops-worker -n "$NAMESPACE"
    
    log_success "Kubernetes deployment completed"
}

# Setup monitoring and alerting
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Install Prometheus using Helm
    if ! helm repo list | grep -q prometheus-community; then
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
    fi
    
    # Install Prometheus stack
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values "$PROJECT_ROOT/k8s/prometheus-values.yaml" \
        --wait
    
    # Apply custom monitoring configurations
    kubectl apply -f "$PROJECT_ROOT/k8s/monitoring-config.yaml"
    
    log_success "Monitoring setup completed"
}

# Run post-deployment validation
validate_deployment() {
    log_info "Validating deployment..."
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE"
    
    # Run health checks
    local api_pod=$(kubectl get pods -n "$NAMESPACE" -l app=finops-api -o jsonpath='{.items[0].metadata.name}')
    if [[ -n "$api_pod" ]]; then
        kubectl exec -n "$NAMESPACE" "$api_pod" -- python scripts/health-check.py --exit-code
    fi
    
    # Check service endpoints
    kubectl get services -n "$NAMESPACE"
    
    log_success "Deployment validation completed"
}

# Main deployment function
main() {
    log_info "Starting production deployment for FinOps Cost Optimization Platform"
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "AWS Region: $AWS_REGION"
    log_info "Cluster: $CLUSTER_NAME"
    log_info "Namespace: $NAMESPACE"
    
    check_prerequisites
    setup_iam_roles
    build_and_push_images
    deploy_to_kubernetes
    setup_monitoring
    validate_deployment
    
    log_success "Production deployment completed successfully!"
    log_info "Access the application at: https://$(kubectl get ingress -n $NAMESPACE -o jsonpath='{.items[0].spec.rules[0].host}')"
    log_info "Monitoring dashboard: https://$(kubectl get ingress -n monitoring -o jsonpath='{.items[0].spec.rules[0].host}')"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "iam-only")
        check_prerequisites
        setup_iam_roles
        ;;
    "build-only")
        check_prerequisites
        build_and_push_images
        ;;
    "k8s-only")
        check_prerequisites
        deploy_to_kubernetes
        ;;
    "monitoring-only")
        check_prerequisites
        setup_monitoring
        ;;
    "validate")
        validate_deployment
        ;;
    *)
        echo "Usage: $0 [deploy|iam-only|build-only|k8s-only|monitoring-only|validate]"
        exit 1
        ;;
esac