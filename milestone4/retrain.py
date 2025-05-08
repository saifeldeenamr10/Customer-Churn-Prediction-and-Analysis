import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_data():
    """Load and prepare data for retraining"""
    try:
        data = pd.read_csv(os.getenv('TRAINING_DATA_PATH', r'C:\Users\Qebaa\OneDrive\Desktop\Graduation\data\processed\data_after_preprocessing.csv'))
        X = data.drop('Exited', axis=1)
        y = data['Exited']
        
        # Convert categorical variables to dummy variables
        X = pd.get_dummies(X, drop_first=True)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def train_model(X_train, y_train):
    """Train a new model"""
    try:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None

def compare_with_production(metrics):
    """Compare new model metrics with production model"""
    try:
        # Get production model metrics
        client = mlflow.tracking.MlflowClient()
        production_run = client.get_latest_versions("churn_model", ["Production"])[0]
        production_metrics = client.get_run(production_run.run_id).data.metrics
        
        # Compare metrics
        improvement = {
            metric: metrics[metric] - production_metrics[metric]
            for metric in metrics.keys()
        }
        
        return improvement
    except Exception as e:
        logger.error(f"Error comparing with production: {e}")
        return None

def promote_model(model, metrics, run_id):
    """Promote model to production if it performs better"""
    try:
        # Compare with production
        improvement = compare_with_production(metrics)
        
        if improvement and all(imp > 0 for imp in improvement.values()):
            # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                "churn_model",
                registered_model_name="churn_model"
            )
            
            # Transition to Production
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="churn_model",
                version=run_id,
                stage="Production"
            )
            
            logger.info("Model promoted to production")
            return True
        else:
            logger.info("Model not promoted - no improvement over production")
            return False
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        return False

def main():
    """Main retraining function"""
    logger.info("Starting model retraining...")
    
    # Load data
    data = load_data()
    if data is None:
        return
    X_train, X_test, y_train, y_test = data
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Train model
        model = train_model(X_train, y_train)
        if model is None:
            return
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        if metrics is None:
            return
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Promote model if better
        promote_model(model, metrics, run.info.run_id)
    
    logger.info("Model retraining completed successfully")

if __name__ == "__main__":
    main() 