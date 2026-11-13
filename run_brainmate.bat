@echo off
title Brainmate - AI Research Assistant Launcher
echo ========================================================
echo        Starting Brainmate AI Research Assistant
echo ========================================================
echo.

echo [1/3] Starting Python Flask ML Service on port 8000...
start "Brainmate - Flask ML Service" cmd /k "python ml_service\app.py"

echo [2/3] Starting Node.js Express Backend on port 5000...
start "Brainmate - Express Backend" cmd /k "cd backend && npm start"

echo [3/3] Starting Next.js Frontend on port 3000...
start "Brainmate - Next.js Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo All services launched!
echo Access Brainmate in your browser at:
echo   http://localhost:3000
echo ========================================================
pause

