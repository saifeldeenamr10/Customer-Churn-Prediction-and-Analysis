# Churn Prediction MLOps Pipeline

This directory contains the MLOps implementation for the churn prediction model, including model deployment, monitoring, and retraining capabilities.

## Components

1. **Model Serving (app.py)**
   - FastAPI-based REST API for model predictions
   - Prometheus metrics for monitoring
   - Health check endpoint
   - Real-time prediction endpoint

2. **Model Monitoring (monitor.py)**
   - Drift detection using Evidently
   - MLflow integration for metric tracking
   - Automated alerts for model drift
   - Performance monitoring

3. **Model Retraining (retrain.py)**
   - Automated model retraining pipeline
   - MLflow integration for experiment tracking
   - Model versioning and promotion
   - Performance comparison with production

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the variables as needed

3. Start MLflow tracking server:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

4. Start the API server:
   ```bash
   python app.py
   ```

## Usage

### Making Predictions
Send POST requests to `/predict` endpoint:
```bash
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"feature1": value1, "feature2": value2, ...}}'
```

### Monitoring
Run the monitoring script:
```bash
python monitor.py
```

### Retraining
Trigger model retraining:
```bash
python retrain.py
```

## Monitoring Dashboard

Access the following dashboards:
- MLflow UI: http://localhost:5000
- Prometheus metrics: http://localhost:8000
- Evidently reports: Generated in the `reports` directory

## Model Retraining Strategy

The model retraining pipeline:
1. Loads new training data
2. Trains a new model
3. Evaluates performance against production
4. Promotes to production if performance improves
5. Logs all metrics and artifacts to MLflow

## Alerting

The system monitors:
- Model drift
- Data drift
- Prediction latency
- Error rates

Alerts are triggered when:
- Drift scores exceed thresholds
- Model performance degrades
- System errors occur

## Best Practices

1. Regularly check MLflow for model performance
2. Monitor drift reports daily
3. Review and adjust drift thresholds as needed
4. Keep training data up to date
5. Document any model changes in MLflow 