@echo off
echo ===================================================
echo   A.R.C.H.E.R. Concurrent Startup Script
echo ===================================================
echo.

:: 1. Start Docker Databases
echo [1/2] Starting database services (Qdrant, Neo4j, Redis, Postgres)...
docker-compose up -d
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start Docker containers. Make sure Docker Desktop is running!
    pause
    exit /b %errorlevel%
)
echo Databases started successfully.
echo.

:: 2. Run Backend & Frontend Concurrently in the Current Terminal
echo [2/2] Booting FastAPI backend and React frontend concurrently in this session...
echo Press Ctrl+C to terminate both servers at any time.
echo.

npx -y concurrently --kill-others -n "backend,frontend" -c "cyan,magenta" ".\venv\Scripts\uvicorn app.main:app --reload --reload-dir app" "npm run dev --prefix frontend"
