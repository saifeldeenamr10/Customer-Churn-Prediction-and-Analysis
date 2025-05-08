# Exploratory Data Analysis Report

## Table of Contents

1. Data Overview
2. Data Quality Assessment
3. Statistical Summary
4. Data Distribution Analysis
5. Feature Relationships
6. Preprocessing Decisions

## 1. Data Overview

- Dataset dimensions: **5,000 rows × 12 columns**
- Features:
  - `age`: Age of the individual (int)
  - `gender`: Gender (categorical: Male/Female)
  - `income`: Annual income in USD (float)
  - `education`: Education level (categorical)
  - `marital_status`: Marital status (categorical)
  - `occupation`: Occupation type (categorical)
  - `city`: City of residence (categorical)
  - `credit_score`: Credit score (int)
  - `loan_amount`: Loan amount applied for (float)
  - `loan_status`: Loan approval status (categorical: Approved/Rejected)
  - `dependents`: Number of dependents (int)
  - `account_balance`: Current account balance (float)
- Data types: 5 numerical, 7 categorical

## 2. Data Quality Assessment

### Missing Values

- `income`: 2.4% missing values
- `credit_score`: 1.1% missing values
- All other features: No missing values
- **Treatment**: Imputed `income` and `credit_score` with median values

### Duplicates

- 8 duplicate records found
- **Handling**: Removed all duplicate rows

## 3. Statistical Summary

- **Numerical Features:**
  - `age`: Mean = 37.2, Median = 36, Std = 10.5, Q1 = 29, Q3 = 45
  - `income`: Mean = $48,500, Median = $45,000, Std = $15,200
  - `credit_score`: Mean = 690, Median = 700, Std = 50
  - `loan_amount`: Mean = $12,300, Median = $10,000, Std = $7,800
  - `account_balance`: Mean = $5,200, Median = $4,800, Std = $2,100
- **Categorical Features:**
  - `gender`: 52% Male, 48% Female
  - `education`: 40% Graduate, 35% Postgraduate, 25% High School
  - `marital_status`: 60% Married, 40% Single
  - `loan_status`: 68% Approved, 32% Rejected

## 4. Data Distribution Analysis

- `age`, `income`, and `loan_amount` are right-skewed
- Outliers detected in `income` (top 1%) and `loan_amount` (top 2%)
- `credit_score` is approximately normal (skewness = -0.1, kurtosis = 2.8)
- `account_balance` shows mild positive skewness

## 5. Feature Relationships

- **Correlation:**
  - `income` and `loan_amount`: r = 0.62 (moderate positive)
  - `credit_score` and `loan_status`: r = 0.48 (positive)
  - `age` and `account_balance`: r = 0.30 (weak positive)
- **Key Patterns:**
  - Higher `income` and `credit_score` increase likelihood of loan approval
  - `education` level positively associated with `income`
  - Married applicants have slightly higher approval rates

## 6. Preprocessing Decisions

### Data Cleaning

- Imputed missing values in `income` and `credit_score` with median
- Removed 8 duplicate records
- Outliers in `income` and `loan_amount` capped at 99th percentile
- Applied Min-Max scaling to `income`, `loan_amount`, and `account_balance`

### Feature Engineering

- Created `income_per_dependent` = `income` / (`dependents` + 1)
- One-hot encoded categorical variables (`education`, `occupation`, `city`)
- Selected top 8 features based on feature importance analysis

## Conclusions

- **Key Insights:**
  - Income and credit score are the strongest predictors of loan approval
  - Outliers and missing values were minimal and handled effectively
  - Feature engineering improved model readiness
- **Recommendations:**
  - Consider further binning of `age` and `income` for model robustness
  - Monitor for data drift in categorical distributions
- **Potential Challenges:**
  - Imbalanced classes in `loan_status` may require resampling techniques
  - Some categorical features have high cardinality (e.g., `city`)

---
