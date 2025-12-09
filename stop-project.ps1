# Cloud Migration Advisor - Shutdown Script
# This script stops all Cloud Migration Advisor services

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cloud Migration Advisor - Stopping..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Stop all services
docker-compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ All services stopped successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To remove all data (volumes), run:" -ForegroundColor Yellow
    Write-Host "  docker-compose down -v" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "✗ Failed to stop services!" -ForegroundColor Red
}
