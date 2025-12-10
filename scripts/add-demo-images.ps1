# PowerShell script to help add demo images to the repository

Write-Host "🎯 Cloud Intelligence FinOps Platform - Demo Images Setup" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Create demo images directory if it doesn't exist
$demoDir = "docs/images/demo"
if (-not (Test-Path $demoDir)) {
    New-Item -ItemType Directory -Path $demoDir -Force
    Write-Host "✅ Created demo images directory: $demoDir" -ForegroundColor Green
}

# List required images
$requiredImages = @(
    "dashboard-overview.png",
    "cost-analysis.png", 
    "migration-wizard.png",
    "migration-dashboard.png",
    "resource-organization.png",
    "provider-recommendations.png",
    "grafana-monitoring.png",
    "api-documentation.png"
)

Write-Host "`n📸 Required Demo Images:" -ForegroundColor Yellow
Write-Host "========================" -ForegroundColor Yellow

foreach ($image in $requiredImages) {
    $imagePath = Join-Path $demoDir $image
    if (Test-Path $imagePath) {
        Write-Host "✅ $image (Found)" -ForegroundColor Green
    } else {
        Write-Host "❌ $image (Missing)" -ForegroundColor Red
    }
}

# Check if platform is running
Write-Host "`n🚀 Platform Status Check:" -ForegroundColor Yellow
Write-Host "=========================" -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Backend API is running (http://localhost:8000)" -ForegroundColor Green
} catch {
    Write-Host "❌ Backend API not accessible. Run: docker-compose up -d" -ForegroundColor Red
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Frontend is running (http://localhost:3000)" -ForegroundColor Green
} catch {
    Write-Host "❌ Frontend not accessible. Run: docker-compose up -d" -ForegroundColor Red
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:3001" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Grafana is running (http://localhost:3001)" -ForegroundColor Green
} catch {
    Write-Host "❌ Grafana not accessible. Run: docker-compose up -d grafana" -ForegroundColor Red
}

# Instructions
Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
Write-Host "==============" -ForegroundColor Yellow
Write-Host "1. Start the platform: docker-compose up -d" -ForegroundColor White
Write-Host "2. Take screenshots of the following URLs:" -ForegroundColor White
Write-Host "   • http://localhost:3000 (Main Dashboard)" -ForegroundColor Cyan
Write-Host "   • http://localhost:3000/cost-analysis (Cost Analysis)" -ForegroundColor Cyan
Write-Host "   • http://localhost:3000/migration-wizard (Migration Wizard)" -ForegroundColor Cyan
Write-Host "   • http://localhost:3000/migration (Migration Dashboard)" -ForegroundColor Cyan
Write-Host "   • http://localhost:3000/resources (Resource Organization)" -ForegroundColor Cyan
Write-Host "   • http://localhost:3000/recommendations (Provider Recommendations)" -ForegroundColor Cyan
Write-Host "   • http://localhost:3001 (Grafana - admin/admin)" -ForegroundColor Cyan
Write-Host "   • http://localhost:8000/docs (API Documentation)" -ForegroundColor Cyan
Write-Host "3. Save images to: $demoDir" -ForegroundColor White
Write-Host "4. Run this script again to verify" -ForegroundColor White
Write-Host "5. Commit and push: git add . && git commit -m 'Add demo images' && git push" -ForegroundColor White

Write-Host "`n🎯 Image Guidelines:" -ForegroundColor Yellow
Write-Host "===================" -ForegroundColor Yellow
Write-Host "• Resolution: 1920x1080 or 1440x900" -ForegroundColor White
Write-Host "• Format: PNG" -ForegroundColor White
Write-Host "• Size: < 2MB each" -ForegroundColor White
Write-Host "• Browser: Chrome/Firefox at 100% zoom" -ForegroundColor White
Write-Host "• Content: Professional, no sensitive data" -ForegroundColor White

Write-Host "`n🌟 After adding images, your demo will be available at:" -ForegroundColor Green
Write-Host "https://github.com/sanketexe/ai-cloud-scheduler/blob/main/docs/DEMO.md" -ForegroundColor Cyan

Write-Host "`n✨ Ready to showcase your amazing platform! 🚀" -ForegroundColor Green