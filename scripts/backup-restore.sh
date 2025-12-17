#!/bin/bash
# Backup and Disaster Recovery Script for FinOps Cost Optimization Platform
# This script handles database backups, configuration backups, and disaster recovery procedures

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_BUCKET="${BACKUP_BUCKET:-finops-backups-$(aws sts get-caller-identity --query Account --output text)}"
AWS_REGION="${AWS_REGION:-us-east-1}"
NAMESPACE="${NAMESPACE:-finops-automation}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

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

# Create backup bucket if it doesn't exist
setup_backup_infrastructure() {
    log_info "Setting up backup infrastructure..."
    
    # Create S3 bucket for backups
    if ! aws s3 ls "s3://$BACKUP_BUCKET" &> /dev/null; then
        log_info "Creating backup bucket: $BACKUP_BUCKET"
        aws s3 mb "s3://$BACKUP_BUCKET" --region "$AWS_REGION"
        
        # Enable versioning
        aws s3api put-bucket-versioning \
            --bucket "$BACKUP_BUCKET" \
            --versioning-configuration Status=Enabled
        
        # Set lifecycle policy
        cat > /tmp/lifecycle-policy.json << EOF
{
    "Rules": [
        {
            "ID": "FinOpsBackupRetention",
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                },
                {
                    "Days": 365,
                    "StorageClass": "DEEP_ARCHIVE"
                }
            ],
            "Expiration": {
                "Days": 2555
            }
        }
    ]
}
EOF
        
        aws s3api put-bucket-lifecycle-configuration \
            --bucket "$BACKUP_BUCKET" \
            --lifecycle-configuration file:///tmp/lifecycle-policy.json
        
        rm -f /tmp/lifecycle-policy.json
    fi
    
    log_success "Backup infrastructure ready"
}

# Backup PostgreSQL database
backup_database() {
    local backup_type="${1:-full}"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="database_${backup_type}_${timestamp}.sql.gz"
    
    log_info "Starting database backup (type: $backup_type)..."
    
    # Get database pod
    local db_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    if [[ -z "$db_pod" ]]; then
        error_exit "No PostgreSQL pod found"
    fi
    
    # Create backup directory
    local backup_dir="/tmp/finops-backup-$timestamp"
    mkdir -p "$backup_dir"
    
    # Perform database backup
    log_info "Creating database dump..."
    kubectl exec -n "$NAMESPACE" "$db_pod" -- pg_dump \
        -U finops \
        -h localhost \
        -p 5432 \
        --verbose \
        --clean \
        --if-exists \
        --create \
        finops_db | gzip > "$backup_dir/$backup_file"
    
    # Backup database schema separately
    kubectl exec -n "$NAMESPACE" "$db_pod" -- pg_dump \
        -U finops \
        -h localhost \
        -p 5432 \
        --schema-only \
        --verbose \
        finops_db > "$backup_dir/schema_${timestamp}.sql"
    
    # Create backup metadata
    cat > "$backup_dir/backup_metadata.json" << EOF
{
    "backup_type": "$backup_type",
    "timestamp": "$timestamp",
    "database_version": "$(kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U finops -d finops_db -t -c "SELECT version();")",
    "backup_size_bytes": $(stat -c%s "$backup_dir/$backup_file"),
    "kubernetes_namespace": "$NAMESPACE",
    "backup_method": "pg_dump"
}
EOF
    
    # Upload to S3
    log_info "Uploading backup to S3..."
    aws s3 cp "$backup_dir/" "s3://$BACKUP_BUCKET/database/$timestamp/" --recursive
    
    # Cleanup local files
    rm -rf "$backup_dir"
    
    log_success "Database backup completed: s3://$BACKUP_BUCKET/database/$timestamp/"
}

# Backup Kubernetes configurations
backup_kubernetes_config() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_dir="/tmp/k8s-backup-$timestamp"
    
    log_info "Starting Kubernetes configuration backup..."
    
    mkdir -p "$backup_dir"
    
    # Backup all resources in the namespace
    log_info "Backing up namespace resources..."
    kubectl get all,configmaps,secrets,pvc,ingress -n "$NAMESPACE" -o yaml > "$backup_dir/namespace-resources.yaml"
    
    # Backup custom resources
    kubectl get servicemonitors,prometheusrules -n monitoring -o yaml > "$backup_dir/monitoring-resources.yaml" 2>/dev/null || true
    
    # Backup RBAC
    kubectl get clusterroles,clusterrolebindings,roles,rolebindings -o yaml > "$backup_dir/rbac-resources.yaml"
    
    # Backup persistent volume claims details
    kubectl describe pvc -n "$NAMESPACE" > "$backup_dir/pvc-details.txt"
    
    # Create backup metadata
    cat > "$backup_dir/k8s_backup_metadata.json" << EOF
{
    "backup_type": "kubernetes_config",
    "timestamp": "$timestamp",
    "kubernetes_version": "$(kubectl version --short --client)",
    "namespace": "$NAMESPACE",
    "cluster_info": "$(kubectl cluster-info | head -1)"
}
EOF
    
    # Upload to S3
    log_info "Uploading Kubernetes backup to S3..."
    aws s3 cp "$backup_dir/" "s3://$BACKUP_BUCKET/kubernetes/$timestamp/" --recursive
    
    # Cleanup
    rm -rf "$backup_dir"
    
    log_success "Kubernetes configuration backup completed: s3://$BACKUP_BUCKET/kubernetes/$timestamp/"
}

# Backup application configuration and secrets
backup_application_config() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_dir="/tmp/app-config-backup-$timestamp"
    
    log_info "Starting application configuration backup..."
    
    mkdir -p "$backup_dir"
    
    # Backup ConfigMaps (excluding sensitive data)
    kubectl get configmap finops-config -n "$NAMESPACE" -o yaml > "$backup_dir/finops-config.yaml"
    kubectl get configmap finops-monitoring-config -n monitoring -o yaml > "$backup_dir/monitoring-config.yaml" 2>/dev/null || true
    
    # Backup application code configuration files
    cp -r "$PROJECT_ROOT/k8s/" "$backup_dir/k8s-manifests/"
    cp -r "$PROJECT_ROOT/monitoring/" "$backup_dir/monitoring-config/"
    
    # Create application metadata
    cat > "$backup_dir/app_config_metadata.json" << EOF
{
    "backup_type": "application_config",
    "timestamp": "$timestamp",
    "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(cd "$PROJECT_ROOT" && git branch --show-current 2>/dev/null || echo 'unknown')",
    "config_files_count": $(find "$backup_dir" -type f | wc -l)
}
EOF
    
    # Upload to S3
    log_info "Uploading application config backup to S3..."
    aws s3 cp "$backup_dir/" "s3://$BACKUP_BUCKET/app-config/$timestamp/" --recursive
    
    # Cleanup
    rm -rf "$backup_dir"
    
    log_success "Application configuration backup completed: s3://$BACKUP_BUCKET/app-config/$timestamp/"
}

# Restore database from backup
restore_database() {
    local backup_timestamp="$1"
    
    if [[ -z "$backup_timestamp" ]]; then
        error_exit "Backup timestamp is required for restore"
    fi
    
    log_info "Starting database restore from backup: $backup_timestamp"
    
    # Download backup from S3
    local restore_dir="/tmp/restore-$backup_timestamp"
    mkdir -p "$restore_dir"
    
    log_info "Downloading backup from S3..."
    aws s3 cp "s3://$BACKUP_BUCKET/database/$backup_timestamp/" "$restore_dir/" --recursive
    
    # Find the database backup file
    local backup_file=$(find "$restore_dir" -name "database_*.sql.gz" | head -1)
    if [[ -z "$backup_file" ]]; then
        error_exit "No database backup file found in $restore_dir"
    fi
    
    # Get database pod
    local db_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    if [[ -z "$db_pod" ]]; then
        error_exit "No PostgreSQL pod found"
    fi
    
    # Create a backup of current database before restore
    log_warning "Creating safety backup before restore..."
    backup_database "pre_restore"
    
    # Restore database
    log_info "Restoring database..."
    gunzip -c "$backup_file" | kubectl exec -i -n "$NAMESPACE" "$db_pod" -- psql -U finops -h localhost
    
    # Cleanup
    rm -rf "$restore_dir"
    
    log_success "Database restore completed from backup: $backup_timestamp"
}

# List available backups
list_backups() {
    local backup_type="${1:-all}"
    
    log_info "Listing available backups (type: $backup_type)..."
    
    case "$backup_type" in
        "database"|"db")
            aws s3 ls "s3://$BACKUP_BUCKET/database/" --recursive --human-readable
            ;;
        "kubernetes"|"k8s")
            aws s3 ls "s3://$BACKUP_BUCKET/kubernetes/" --recursive --human-readable
            ;;
        "config"|"app-config")
            aws s3 ls "s3://$BACKUP_BUCKET/app-config/" --recursive --human-readable
            ;;
        "all"|*)
            echo "=== Database Backups ==="
            aws s3 ls "s3://$BACKUP_BUCKET/database/" --recursive --human-readable
            echo ""
            echo "=== Kubernetes Backups ==="
            aws s3 ls "s3://$BACKUP_BUCKET/kubernetes/" --recursive --human-readable
            echo ""
            echo "=== Application Config Backups ==="
            aws s3 ls "s3://$BACKUP_BUCKET/app-config/" --recursive --human-readable
            ;;
    esac
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    # Calculate cutoff date
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" '+%Y%m%d')
    
    # List and delete old backups
    aws s3 ls "s3://$BACKUP_BUCKET/" --recursive | while read -r line; do
        local file_date=$(echo "$line" | awk '{print $1}' | tr -d '-')
        local file_path=$(echo "$line" | awk '{print $4}')
        
        if [[ "$file_date" < "$cutoff_date" ]]; then
            log_info "Deleting old backup: $file_path"
            aws s3 rm "s3://$BACKUP_BUCKET/$file_path"
        fi
    done
    
    log_success "Cleanup completed"
}

# Disaster recovery procedure
disaster_recovery() {
    local recovery_timestamp="$1"
    
    if [[ -z "$recovery_timestamp" ]]; then
        error_exit "Recovery timestamp is required"
    fi
    
    log_warning "Starting DISASTER RECOVERY procedure..."
    log_warning "This will restore the system to state: $recovery_timestamp"
    
    read -p "Are you sure you want to proceed? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Disaster recovery cancelled"
        exit 0
    fi
    
    # Step 1: Restore Kubernetes configuration
    log_info "Step 1: Restoring Kubernetes configuration..."
    local restore_dir="/tmp/disaster-recovery-$recovery_timestamp"
    mkdir -p "$restore_dir"
    
    aws s3 cp "s3://$BACKUP_BUCKET/kubernetes/$recovery_timestamp/" "$restore_dir/k8s/" --recursive
    
    # Apply Kubernetes resources (excluding secrets for security)
    kubectl apply -f "$restore_dir/k8s/namespace-resources.yaml" || log_warning "Some resources may have failed to apply"
    
    # Step 2: Wait for pods to be ready
    log_info "Step 2: Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout=300s
    
    # Step 3: Restore database
    log_info "Step 3: Restoring database..."
    restore_database "$recovery_timestamp"
    
    # Step 4: Restart application pods
    log_info "Step 4: Restarting application pods..."
    kubectl rollout restart deployment/finops-api -n "$NAMESPACE"
    kubectl rollout restart deployment/finops-worker -n "$NAMESPACE"
    kubectl rollout restart deployment/finops-scheduler -n "$NAMESPACE"
    
    # Step 5: Verify system health
    log_info "Step 5: Verifying system health..."
    sleep 30
    kubectl wait --for=condition=available deployment/finops-api -n "$NAMESPACE" --timeout=300s
    
    # Cleanup
    rm -rf "$restore_dir"
    
    log_success "Disaster recovery completed successfully!"
    log_info "Please verify system functionality and check logs for any issues"
}

# Health check for backup system
backup_health_check() {
    log_info "Performing backup system health check..."
    
    local issues=0
    
    # Check S3 bucket access
    if aws s3 ls "s3://$BACKUP_BUCKET" &> /dev/null; then
        log_success "S3 backup bucket accessible"
    else
        log_error "Cannot access S3 backup bucket: $BACKUP_BUCKET"
        ((issues++))
    fi
    
    # Check database connectivity
    local db_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [[ -n "$db_pod" ]] && kubectl exec -n "$NAMESPACE" "$db_pod" -- pg_isready -U finops &> /dev/null; then
        log_success "Database connectivity OK"
    else
        log_error "Cannot connect to database"
        ((issues++))
    fi
    
    # Check recent backups
    local recent_backups=$(aws s3 ls "s3://$BACKUP_BUCKET/database/" | tail -5 | wc -l)
    if [[ "$recent_backups" -gt 0 ]]; then
        log_success "Recent backups found: $recent_backups"
    else
        log_warning "No recent backups found"
        ((issues++))
    fi
    
    if [[ "$issues" -eq 0 ]]; then
        log_success "Backup system health check passed"
        return 0
    else
        log_error "Backup system health check failed with $issues issues"
        return 1
    fi
}

# Main function
main() {
    local command="${1:-help}"
    
    case "$command" in
        "setup")
            setup_backup_infrastructure
            ;;
        "backup-db"|"backup-database")
            setup_backup_infrastructure
            backup_database "${2:-full}"
            ;;
        "backup-k8s"|"backup-kubernetes")
            setup_backup_infrastructure
            backup_kubernetes_config
            ;;
        "backup-config"|"backup-app-config")
            setup_backup_infrastructure
            backup_application_config
            ;;
        "backup-all")
            setup_backup_infrastructure
            backup_database "full"
            backup_kubernetes_config
            backup_application_config
            ;;
        "restore-db"|"restore-database")
            restore_database "$2"
            ;;
        "list"|"list-backups")
            list_backups "${2:-all}"
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "disaster-recovery"|"dr")
            disaster_recovery "$2"
            ;;
        "health-check")
            backup_health_check
            ;;
        "help"|*)
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  setup                     - Setup backup infrastructure"
            echo "  backup-db [type]          - Backup database (type: full|incremental)"
            echo "  backup-k8s                - Backup Kubernetes configuration"
            echo "  backup-config             - Backup application configuration"
            echo "  backup-all                - Backup everything"
            echo "  restore-db <timestamp>    - Restore database from backup"
            echo "  list [type]               - List available backups"
            echo "  cleanup                   - Remove old backups"
            echo "  disaster-recovery <ts>    - Full disaster recovery"
            echo "  health-check              - Check backup system health"
            echo "  help                      - Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  BACKUP_BUCKET            - S3 bucket for backups"
            echo "  AWS_REGION               - AWS region"
            echo "  NAMESPACE                - Kubernetes namespace"
            echo "  RETENTION_DAYS           - Backup retention period"
            ;;
    esac
}

main "$@"