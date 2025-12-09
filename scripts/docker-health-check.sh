#!/bin/bash
# Docker Health Check Script for FinOps Platform
# This script is used by Docker health checks to verify container health

set -e

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-/health/ready}"
TIMEOUT="${TIMEOUT:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Health check function
check_health() {
    local url="$1"
    local timeout="$2"
    
    # Use curl if available, otherwise use wget
    if command -v curl >/dev/null 2>&1; then
        response=$(curl -s -w "%{http_code}" --max-time "$timeout" "$url" || echo "000")
        http_code="${response: -3}"
        body="${response%???}"
    elif command -v wget >/dev/null 2>&1; then
        if wget --timeout="$timeout" --tries=1 -q -O - "$url" >/dev/null 2>&1; then
            http_code="200"
            body=""
        else
            http_code="000"
            body=""
        fi
    else
        log "ERROR: Neither curl nor wget is available"
        return 1
    fi
    
    if [ "$http_code" = "200" ]; then
        log "Health check passed (HTTP $http_code)"
        return 0
    else
        log "Health check failed (HTTP $http_code)"
        return 1
    fi
}

# Main health check with retries
main() {
    local url="$API_URL$HEALTH_ENDPOINT"
    local retry_count=0
    
    log "Starting health check for $url"
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if check_health "$url" "$TIMEOUT"; then
            echo -e "${GREEN}✓ Health check successful${NC}"
            exit 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log "Retry $retry_count/$MAX_RETRIES in 2 seconds..."
                sleep 2
            fi
        fi
    done
    
    echo -e "${RED}✗ Health check failed after $MAX_RETRIES attempts${NC}"
    exit 1
}

# Handle different service types
case "${SERVICE_TYPE:-api}" in
    "api")
        HEALTH_ENDPOINT="/health/ready"
        ;;
    "worker")
        # For worker, check if the process is running
        if pgrep -f "celery.*worker" >/dev/null; then
            echo -e "${GREEN}✓ Worker process is running${NC}"
            exit 0
        else
            echo -e "${RED}✗ Worker process not found${NC}"
            exit 1
        fi
        ;;
    "scheduler")
        # For scheduler, check if the process is running and beat file exists
        if pgrep -f "celery.*beat" >/dev/null && [ -f "/tmp/celerybeat.pid" ]; then
            echo -e "${GREEN}✓ Scheduler process is running${NC}"
            exit 0
        else
            echo -e "${RED}✗ Scheduler process not found or beat file missing${NC}"
            exit 1
        fi
        ;;
    "frontend")
        API_URL="http://localhost"
        HEALTH_ENDPOINT="/"
        ;;
    *)
        log "Unknown service type: ${SERVICE_TYPE}"
        exit 1
        ;;
esac

main