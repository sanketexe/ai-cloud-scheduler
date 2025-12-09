# Cloud Migration Advisor - Startup Script
# This script starts the entire Cloud Migration Advisor platform

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cloud Migration Advisor - Starting..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Docker Desktop from:" -ForegroundColor Yellow
    Write-Host "https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Check if Docker is running
try {
    docker ps | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Starting services..." -ForegroundColor Cyan

# Start all services
docker-compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Services started successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access the services at:" -ForegroundColor Cyan
    Write-Host "  • API:              http://localhost:8000" -ForegroundColor White
    Write-Host "  • API Docs:         http://localhost:8000/docs" -ForegroundColor White
    Write-Host "  • Frontend:         http://localhost:3000" -ForegroundColor White
    Write-Host "  • Grafana:          http://localhost:3001 (admin/admin)" -ForegroundColor White
    Write-Host "  • Prometheus:       http://localhost:9090" -ForegroundColor White
    Write-Host "  • Kibana:           http://localhost:5601" -ForegroundColor White
    Write-Host ""
    Write-Host "View logs with:" -ForegroundColor Cyan
    Write-Host "  docker-compose logs -f api" -ForegroundColor White
    Write-Host ""
    Write-Host "Stop services with:" -ForegroundColor Cyan
    Write-Host "  docker-compose down" -ForegroundColor White
    Write-Host ""
    
    # Wait for services to be healthy
    Write-Host "Waiting for services to be ready..." -ForegroundColor Cyan
    Start-Sleep -Seconds 10
    
    # Check API health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ API is healthy and ready!" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠ API is starting up... (this may take a minute)" -ForegroundColor Yellow
    }
    
} else {
    Write-Host ""
    Write-Host "✗ Failed to start services!" -ForegroundColor Red
    Write-Host "Check the logs with: docker-compose logs" -ForegroundColor Yellow
}
