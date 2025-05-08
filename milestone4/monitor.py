import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from evidently.ui.dashboards import CounterAgg, DashboardPanelCounter
from evidently.ui.workspace import Workspace
import mlflow
import os
from datetime import datetime
import logging
from dotenv import load_dotenv
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define file paths
DATA_PATH = r"C:\Users\Qebaa\OneDrive\Desktop\Graduation\data\processed\data_after_preprocessing.csv"
MODEL_PATH = r"C:\Users\Qebaa\OneDrive\Desktop\Graduation\milestone3\models\best_model_Logistic_Regression.pkl"

# Create monitoring output directory
MONITORING_DIR = Path("milestone4/monitoring_output")
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

# Configure MLflow
os.environ['MLFLOW_TRACKING_URI'] = str(MONITORING_DIR / 'mlruns')
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

def setup_mlflow_experiment():
    """Setup MLflow experiment"""
    experiment_name = "model_monitoring"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def load_data():
    """Load the current data"""
    try:
        data = pd.read_csv(DATA_PATH)
        # Prepare features (excluding target)
        X = pd.get_dummies(data.drop('Exited', axis=1), drop_first=True)
        y = data['Exited']
        return X, y, data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None

def generate_predictions(model, X):
    """Generate predictions using the loaded model"""
    try:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        return predictions, probabilities
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return None, None

def create_monitoring_visualizations(reference_data, current_data, predictions, raw_data, model):
    """Create and save monitoring visualizations"""
    try:
        # Create visualization directory
        viz_dir = MONITORING_DIR / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Feature Distribution Comparison
        for column in reference_data.columns:
            if column not in ['target', 'prediction']:
                plt.figure(figsize=(10, 6))
                
                # Handle different data types appropriately
                if reference_data[column].dtype == bool:
                    ref_data = reference_data[column].astype(int)
                    curr_data = current_data[column].astype(int)
                    plt.hist(ref_data, alpha=0.5, label='Reference', bins=2)
                    plt.hist(curr_data, alpha=0.5, label='Current', bins=2)
                elif np.issubdtype(reference_data[column].dtype, np.number):
                    plt.hist(reference_data[column], alpha=0.5, label='Reference', bins=30)
                    plt.hist(current_data[column], alpha=0.5, label='Current', bins=30)
                else:
                    # For categorical data, use bar plots
                    ref_counts = reference_data[column].value_counts()
                    curr_counts = current_data[column].value_counts()
                    
                    # Get all unique categories
                    all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
                    x = np.arange(len(all_categories))
                    width = 0.35
                    
                    plt.bar(x - width/2, [ref_counts.get(cat, 0) for cat in all_categories], 
                           width, label='Reference', alpha=0.5)
                    plt.bar(x + width/2, [curr_counts.get(cat, 0) for cat in all_categories], 
                           width, label='Current', alpha=0.5)
                    plt.xticks(x, all_categories, rotation=45)
                
                plt.title(f'Distribution Comparison: {column}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(viz_dir / f'dist_comparison_{column}.png')
                plt.close()
        
        # 2. Prediction Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=predictions.astype(int), label='Predictions', bins=2)
        plt.title('Prediction Distribution')
        plt.xlabel('Prediction (0: Not Exited, 1: Exited)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / 'prediction_distribution.png')
        plt.close()
        
        # 3. Feature Correlation Matrix for Numeric Features
        numeric_data = raw_data.select_dtypes(include=[np.number, bool]).copy()
        for column in numeric_data.columns:
            if numeric_data[column].dtype == bool:
                numeric_data[column] = numeric_data[column].astype(int)
        
        plt.figure(figsize=(12, 8))
        correlation_matrix = numeric_data.corr()
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(correlation_matrix,
                   mask=mask,
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f',
                   square=True)
        plt.title('Feature Correlation Matrix (Numeric Features)')
        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_matrix.png')
        plt.close()
        
        # 4. Model Performance Visualization
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(current_data['target'], current_data['prediction'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(viz_dir / 'confusion_matrix.png')
        plt.close()
        
        # 5. Feature Importance Plot (for numeric features)
        try:
            from sklearn.inspection import permutation_importance
            numeric_features = raw_data.select_dtypes(include=[np.number, bool]).columns
            X_numeric = raw_data[numeric_features].copy()
            for col in X_numeric.columns:
                if X_numeric[col].dtype == bool:
                    X_numeric[col] = X_numeric[col].astype(int)
            
            result = permutation_importance(model, X_numeric, raw_data['Exited'], n_repeats=10, random_state=42)
            importance = pd.DataFrame({
                'feature': numeric_features,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 6))
            plt.barh(importance['feature'], importance['importance'])
            plt.title('Feature Importance (Numeric Features)')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(viz_dir / 'feature_importance.png')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {e}")
        
        return viz_dir
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        logger.error(f"Error details: {str(e)}")
        return None

def generate_drift_report(reference_data, current_data):
    """Generate drift report using Evidently"""
    try:
        data_drift = DataDriftPreset()
        target_drift = TargetDriftPreset()
        dataset_drift = DatasetDriftMetric()
        column_drift = ColumnDriftMetric(column_name="prediction")
        
        report = Report(metrics=[
            data_drift,
            target_drift,
            dataset_drift,
            column_drift
        ])
        
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Save detailed report
        report_path = MONITORING_DIR / 'reports' / f'drift_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        report_path.parent.mkdir(exist_ok=True)
        report.save_html(str(report_path))
        
        return report, data_drift, target_drift, dataset_drift, column_drift
    except Exception as e:
        logger.error(f"Error generating drift report: {e}")
        return None, None, None, None, None

def log_metrics_to_mlflow(report, metrics, model_metrics, viz_dir):
    """Log drift metrics and model performance metrics to MLflow"""
    try:
        with mlflow.start_run(run_name=f"model_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            data_drift, target_drift, dataset_drift, column_drift = metrics
            
            # Log drift metrics
            if dataset_drift and hasattr(dataset_drift, 'result') and dataset_drift.result and hasattr(dataset_drift.result, 'dataset_drift'):
                mlflow.log_metric("dataset_drift", float(dataset_drift.result.dataset_drift))
            
            # Log feature drift scores if available
            if data_drift and hasattr(data_drift, 'result') and data_drift.result:
                drift_scores = data_drift.result.drift_scores
                if drift_scores:
                    for feature, score in drift_scores.items():
                        mlflow.log_metric(f"drift_score_{feature}", float(score))
            
            # Log model performance metrics
            for metric_name, value in model_metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log visualizations
            if viz_dir:
                for viz_file in viz_dir.glob('*.png'):
                    mlflow.log_artifact(str(viz_file))
            
            # Log report
            report_path = next(MONITORING_DIR.glob('reports/*.html'))
            if report_path.exists():
                mlflow.log_artifact(str(report_path))
            
    except Exception as e:
        logger.error(f"Error logging metrics to MLflow: {e}")

def check_drift_thresholds(metrics):
    """Check if drift exceeds thresholds and trigger alerts"""
    try:
        drift_threshold = 0.3  # You can adjust this threshold
        data_drift, target_drift, dataset_drift, column_drift = metrics
        
        if dataset_drift and hasattr(dataset_drift, 'result') and dataset_drift.result and hasattr(dataset_drift.result, 'dataset_drift'):
            drift_score = float(dataset_drift.result.dataset_drift)
            if drift_score > drift_threshold:
                logger.warning(f"Dataset drift detected! Score: {drift_score:.4f}")
        
        if target_drift and hasattr(target_drift, 'result') and target_drift.result and hasattr(target_drift.result, 'target_drift'):
            target_drift_score = float(target_drift.result.target_drift)
            if target_drift_score > drift_threshold:
                logger.warning(f"Target drift detected! Score: {target_drift_score:.4f}")
    except Exception as e:
        logger.error(f"Error checking drift thresholds: {e}")

def generate_summary_report(model_metrics, drift_metrics, raw_data):
    """Generate a summary report in markdown format"""
    try:
        summary_path = MONITORING_DIR / 'reports' / f'summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        
        with open(summary_path, 'w') as f:
            f.write("# Model Monitoring Summary Report\n\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data Overview
            f.write("## Data Overview\n")
            f.write(f"- Total samples: {len(raw_data)}\n")
            f.write(f"- Features: {len(raw_data.columns) - 1}\n")  # Excluding target
            f.write("- Data types:\n")
            for dtype in raw_data.dtypes.value_counts().items():
                f.write(f"  - {dtype[0]}: {dtype[1]} features\n")
            
            # Model Performance
            f.write("\n## Model Performance Metrics\n")
            for metric_name, value in model_metrics.items():
                f.write(f"- {metric_name}: {value:.4f}\n")
            
            # Data Drift Analysis
            f.write("\n## Data Drift Analysis\n")
            if drift_metrics:
                data_drift, target_drift, dataset_drift, column_drift = drift_metrics
                if dataset_drift and hasattr(dataset_drift, 'result') and dataset_drift.result:
                    f.write(f"- Overall Dataset Drift: {dataset_drift.result.dataset_drift:.4f}\n")
                if target_drift and hasattr(target_drift, 'result') and target_drift.result:
                    f.write(f"- Target Drift: {target_drift.result.target_drift:.4f}\n")
            
            # Class Distribution
            f.write("\n## Class Distribution\n")
            class_dist = raw_data['Exited'].value_counts()
            f.write(f"- Not Exited (0): {class_dist[0]} ({class_dist[0]/len(raw_data)*100:.2f}%)\n")
            f.write(f"- Exited (1): {class_dist[1]} ({class_dist[1]/len(raw_data)*100:.2f}%)\n")
            
            # Recommendations
            f.write("\n## Recommendations\n")
            if model_metrics['accuracy'] < 0.95:
                f.write("- Model accuracy has dropped below 95%. Consider retraining.\n")
            if model_metrics['f1'] < 0.95:
                f.write("- F1 score is below threshold. Check for class imbalance issues.\n")
            if class_dist[1]/len(raw_data) < 0.1:
                f.write("- Severe class imbalance detected. Consider using balanced training techniques.\n")
            
            # Visualization Guide
            f.write("\n## Available Visualizations\n")
            f.write("The following visualizations have been generated:\n")
            f.write("1. Feature Distribution Comparisons\n")
            f.write("2. Prediction Distribution\n")
            f.write("3. Feature Correlation Matrix\n")
            f.write("4. Confusion Matrix\n")
            f.write("5. Feature Importance Plot\n")
            
        return summary_path
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return None

def main():
    """Main monitoring function"""
    logger.info("Starting model monitoring...")
    
    # Setup MLflow
    setup_mlflow_experiment()
    
    # Load model
    model = load_model()
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Load data
    X, y, raw_data = load_data()
    if X is None or y is None:
        logger.error("Failed to load data")
        return
    
    # Generate predictions
    predictions, probabilities = generate_predictions(model, X)
    if predictions is None:
        logger.error("Failed to generate predictions")
        return
    
    # Prepare data for monitoring
    monitoring_data = X.copy()
    monitoring_data['target'] = y
    monitoring_data['prediction'] = predictions
    
    # Split data into reference and current (for demonstration)
    from sklearn.model_selection import train_test_split
    reference_data, current_data = train_test_split(monitoring_data, test_size=0.3, random_state=42)
    
    # Generate drift report
    report_results = generate_drift_report(reference_data, current_data)
    if report_results[0] is None:  # First element is the report
        logger.error("Failed to generate drift report")
        return
    
    report, *metrics = report_results
    
    # Create visualizations
    viz_dir = create_monitoring_visualizations(reference_data, current_data, predictions, raw_data, model)
    
    # Calculate model metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    model_metrics = {
        'accuracy': accuracy_score(y, predictions),
        'roc_auc': roc_auc_score(y, probabilities[:, 1]),
        'precision': precision_score(y, predictions),
        'recall': recall_score(y, predictions),
        'f1': f1_score(y, predictions)
    }
    
    # Generate summary report
    summary_path = generate_summary_report(model_metrics, metrics, raw_data)
    
    # Log metrics to MLflow
    log_metrics_to_mlflow(report, metrics, model_metrics, viz_dir)
    
    # Check drift thresholds
    check_drift_thresholds(metrics)
    
    logger.info("Model monitoring completed successfully")
    logger.info("\nModel Performance Metrics:")
    for metric_name, value in model_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    logger.info(f"\nMonitoring artifacts saved to: {MONITORING_DIR}")
    if summary_path:
        logger.info(f"Summary report generated at: {summary_path}")

if __name__ == "__main__":
    main() 