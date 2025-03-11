# Telco Customer Churn Preprocessing

This repository contains a Jupyter Notebook for preprocessing the Telco Customer Churn dataset to prepare it for machine learning tasks focused on churn prediction.

## Overview
- **Data**: Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Tasks**: 
  - Exploration: Checks for missing values, duplicates, and outliers.
  - Preprocessing: Encodes categorical variables, scales numerical features, and handles special cases (e.g., 'No internet service').
  - Feature Engineering: Adds `SpendingPerMonth` feature.
  - Output: Saves cleaned datasets as CSVs.

## Requirements
- Python 3.x
- Dependencies: Listed in `requirements.txt`

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/telco-churn-preprocessing.git
