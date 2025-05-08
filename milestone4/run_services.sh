#!/bin/bash

# Function to check if a port is in use
check_port() {
    if netstat -tuln | grep -q ":$1 "; then
        return 0
    else
        return 1
    fi
}

echo "Starting MLOps services..."

# Start MLflow server if not running
if ! check_port 5000; then
    echo "Starting MLflow server..."
    mlflow server --host 0.0.0.0 --port 5000 &
    sleep 5
else
    echo "MLflow server already running on port 5000"
fi

# Start FastAPI application if not running
if ! check_port 8001; then
    echo "Starting prediction API..."
    python app.py &
    sleep 5
else
    echo "Prediction API already running on port 8001"
fi

# Run monitoring
echo "Running model monitoring..."
python monitor.py

echo "All services started. Use the following endpoints:"
echo "- MLflow UI: http://localhost:5000"
echo "- Prediction API: http://localhost:8001"
echo "- Prometheus metrics: http://localhost:8000"
echo "- Monitoring reports: Check milestone4/monitoring_output/reports/" 