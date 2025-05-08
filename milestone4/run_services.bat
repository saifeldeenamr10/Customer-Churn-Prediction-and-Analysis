@echo off
echo Starting MLOps services...

REM Check if MLflow server is running
netstat -ano | findstr ":5000" > nul
if errorlevel 1 (
    echo Starting MLflow server...
    start "MLflow Server" mlflow server --host 0.0.0.0 --port 5000
    timeout /t 5 /nobreak > nul
) else (
    echo MLflow server already running on port 5000
)

REM Check if FastAPI application is running
netstat -ano | findstr ":8001" > nul
if errorlevel 1 (
    echo Starting prediction API...
    start "Prediction API" python app.py
    timeout /t 5 /nobreak > nul
) else (
    echo Prediction API already running on port 8001
)

REM Run monitoring
echo Running model monitoring...
python monitor.py

echo.
echo All services started. Use the following endpoints:
echo - MLflow UI: http://localhost:5000
echo - Prediction API: http://localhost:8001
echo - Prometheus metrics: http://localhost:8000
echo - Monitoring reports: Check milestone4/monitoring_output/reports/
echo.
pause 