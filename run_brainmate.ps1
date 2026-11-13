# Brainmate AI Research Assistant - PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       Starting Brainmate AI Research Assistant" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$root = $PSScriptRoot

Write-Host "`n[1/3] Starting Python Flask ML Service on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; python ml_service\app.py"

Write-Host "[2/3] Starting Node.js Express Backend on port 5000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\backend'; npm start"

Write-Host "[3/3] Starting Next.js Frontend on port 3000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\frontend'; npm run dev"

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "All 3 services launched successfully!" -ForegroundColor Green
Write-Host "Open your browser to: http://localhost:3000" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green

