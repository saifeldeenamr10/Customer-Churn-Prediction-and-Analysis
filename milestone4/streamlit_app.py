import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🎯 Customer Churn Prediction Dashboard")
st.markdown("""
This dashboard provides an interface to:
- Make real-time churn predictions
- View model performance metrics
- Monitor data and prediction drift
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Make Prediction", "Model Performance", "Monitoring"])

if page == "Make Prediction":
    st.header("Make a Prediction")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
            balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0)
            num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
            
        with col2:
            has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)
            complain = st.selectbox("Has Complaints?", ["Yes", "No"])
            satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
            
        with col3:
            card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
            points_earned = st.number_input("Points Earned", min_value=0, max_value=1000, value=500)
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        submitted = st.form_submit_button("Predict")
        
    if submitted:
        # Prepare features
        features = {
            "CreditScore": float(credit_score),
            "Age": float(age),
            "Tenure": float(tenure),
            "Balance": float(balance),
            "NumOfProducts": float(num_products),
            "HasCrCard": has_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": float(estimated_salary),
            "Complain": complain,
            "Satisfaction Score": float(satisfaction_score),
            "Card Type": card_type,
            "Point Earned": float(points_earned),
            "BalanceToSalary": float(balance / estimated_salary if estimated_salary > 0 else 0),
            "ProductsPerTenure": float(num_products / tenure if tenure > 0 else num_products),
            "Geography_Germany": float(1 if geography == "Germany" else 0),
            "Geography_Spain": float(1 if geography == "Spain" else 0),
            "Gender_Male": float(1 if gender == "Male" else 0)
        }
        
        # Make prediction request
        try:
            with st.spinner('Making prediction...'):
                response = requests.post(
                    "http://localhost:8001/predict",
                    json={"features": features},
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display prediction
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Prediction", 
                        "Will Churn" if result["prediction"] == 1 else "Will Stay",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Churn Probability", 
                        f"{result['probability']:.2%}",
                        delta=None
                    )
                
                # Add visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result['probability'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Risk"},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgreen"},
                            {'range': [0.3, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.7
                        }
                    }
                ))
                st.plotly_chart(fig)
                
            elif response.status_code == 500:
                st.error("The prediction service is not ready. Please make sure the model is properly trained and the server is running.")
                st.info("Try running 'python train_and_save.py' to train the model first.")
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"Error making prediction: {error_detail}")
                if isinstance(error_detail, list):
                    for error in error_detail:
                        st.error(f"- {error.get('msg', '')}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the prediction service. Please make sure the FastAPI server is running on http://localhost:8001")
            st.info("Run 'uvicorn app:app --host localhost --port 8001' to start the server.")
        except requests.exceptions.Timeout:
            st.error("The prediction service took too long to respond. Please try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Load monitoring data
    try:
        monitoring_dir = "monitoring_output/reports"
        latest_report = max([f for f in os.listdir(monitoring_dir) if f.startswith('summary_report')], 
                          key=lambda x: os.path.getctime(os.path.join(monitoring_dir, x)))
        
        with open(os.path.join(monitoring_dir, latest_report), 'r') as f:
            report_content = f.read()
            
        # Display metrics
        metrics_section = report_content.split("## Model Performance Metrics")[1].split("##")[0]
        metrics = {}
        for line in metrics_section.strip().split("\n"):
            if line.startswith("- "):
                metric, value = line[2:].split(": ")
                metrics[metric] = float(value)
        
        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.4f}")
            
        # Add performance visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error loading model performance data: {str(e)}")

else:  # Monitoring page
    st.header("Model and Data Monitoring")
    
    try:
        # Display monitoring visualizations
        st.subheader("Monitoring Visualizations")
        
        viz_dir = "monitoring_output/visualizations"
        
        # Display confusion matrix
        if os.path.exists(os.path.join(viz_dir, "confusion_matrix.png")):
            st.image(os.path.join(viz_dir, "confusion_matrix.png"), 
                    caption="Confusion Matrix",
                    use_column_width=True)
        
        # Display feature importance
        if os.path.exists(os.path.join(viz_dir, "feature_importance.png")):
            st.image(os.path.join(viz_dir, "feature_importance.png"), 
                    caption="Feature Importance",
                    use_column_width=True)
        
        # Display correlation matrix
        if os.path.exists(os.path.join(viz_dir, "correlation_matrix.png")):
            st.image(os.path.join(viz_dir, "correlation_matrix.png"), 
                    caption="Feature Correlation Matrix",
                    use_column_width=True)
            
        # Load and display drift report
        monitoring_dir = "monitoring_output/reports"
        latest_report = max([f for f in os.listdir(monitoring_dir) if f.startswith('summary_report')], 
                          key=lambda x: os.path.getctime(os.path.join(monitoring_dir, x)))
        
        with open(os.path.join(monitoring_dir, latest_report), 'r') as f:
            report_content = f.read()
        
        st.subheader("Latest Monitoring Report")
        st.markdown(report_content)
        
    except Exception as e:
        st.error(f"Error loading monitoring data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 