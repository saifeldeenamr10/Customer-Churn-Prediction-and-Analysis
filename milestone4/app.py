from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from prometheus_client import Counter, Histogram, start_http_server
import time
import os
from typing import Union, Dict

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API")

# Prometheus metrics
PREDICTION_COUNTER = Counter('prediction_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')

# Load model and encoders
try:
    model = joblib.load(os.path.join(MODELS_DIR, 'churn_model.joblib'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
    card_type_encoder = joblib.load(os.path.join(MODELS_DIR, 'card_type_encoder.joblib'))
    print("Model and encoders loaded successfully!")
    print("Available features:", feature_names)
except Exception as e:
    print(f"Error loading model and encoders: {e}")
    model = None
    feature_names = None
    card_type_encoder = None

class PredictionInput(BaseModel):
    features: Dict[str, Union[float, str, int]]

@app.get("/")
async def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if model is None or feature_names is None or card_type_encoder is None:
        raise HTTPException(status_code=500, detail="Model or encoders not loaded")
    
    start_time = time.time()
    
    try:
        # Process features
        features_dict = input_data.features.copy()
        
        # Convert binary features from Yes/No to 1/0
        binary_features = ['HasCrCard', 'IsActiveMember', 'Complain']
        for feature in binary_features:
            if feature in features_dict:
                if isinstance(features_dict[feature], str):
                    features_dict[feature] = 1 if features_dict[feature].lower() == 'yes' else 0
                else:
                    features_dict[feature] = int(bool(features_dict[feature]))
        
        # Encode Card Type
        if 'Card Type' in features_dict:
            try:
                card_type = str(features_dict['Card Type']).upper()
                features_dict['Card Type'] = card_type_encoder.transform([card_type])[0]
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid Card Type value '{features_dict['Card Type']}'. Must be one of: {list(card_type_encoder.classes_)}"
                )
        
        # Convert all values to float
        try:
            for feature in features_dict:
                if feature != 'Card Type':  # Skip Card Type as it's already encoded
                    features_dict[feature] = float(features_dict[feature])
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error converting feature values to numeric: {str(e)}"
            )
        
        # Create feature array in correct order
        try:
            feature_array = np.array([[features_dict[feature] for feature in feature_names]])
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature: {str(e)}"
            )
        
        # Make prediction
        prediction = model.predict(feature_array)
        probability = model.predict_proba(feature_array)
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1]),
            "status": "success"
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_names": feature_names is not None,
        "encoders_loaded": card_type_encoder is not None
    }

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001) 