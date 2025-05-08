# Data Analysis Report

## Statistical Analysis and Insights

This section summarizes the statistical tests and insights derived from the advanced data analysis notebook:

- **T-tests**: Conducted on numerical features to compare means between churned and non-churned customers. Features like `CreditScore` and `Balance` showed statistically significant differences, indicating their potential predictive power.
- **Chi-squared Tests**: Performed on categorical features such as `Geography` and `Gender` to evaluate their relationship with churn. Results highlighted that `Geography` had a significant association with churn rates.
- **Correlation Matrix**: Visualized relationships between numerical features to identify multicollinearity and feature importance. High correlations were observed between `Balance` and `EstimatedSalary`, suggesting potential redundancy in predictive modeling.
- **Recursive Feature Elimination (RFE)**: Identified the top 5 most relevant features for churn prediction, including `Tenure`, `BalanceToSalary`, and `ProductsPerTenure`. These features were selected based on their contribution to model performance.

### Key Insights:

1. Features such as `CreditScore`, `Balance`, and `EstimatedSalary` showed significant differences between churned and non-churned groups, with churned customers generally having lower `CreditScore` and higher `Balance`.
2. Strong correlations were observed between `Balance` and `EstimatedSalary`, suggesting potential redundancy and the need for dimensionality reduction techniques.
3. RFE highlighted features like `Tenure`, `BalanceToSalary`, and `ProductsPerTenure` as critical predictors, emphasizing the importance of customer lifecycle and financial behavior in churn prediction.
4. Customers with higher `ProductsPerTenure` ratios were less likely to churn, indicating that engagement over time plays a key role in retention.

---

## Feature Engineering Summary

This section outlines the new features, transformations, and their expected impact on model performance from the feature engineering notebook:

### New Features:

1. **Tenure (in months)**: Derived from `last_active_date` and `signup_date`. This feature captures the duration of a customer's relationship with the company and is expected to improve the model's ability to capture customer lifecycle patterns.
2. **BalanceToSalary Ratio**: Created to measure financial stability by dividing `Balance` by `EstimatedSalary`. This feature is anticipated to enhance the model's predictive power for financial behavior.
3. **ProductsPerTenure**: Calculated as the ratio of products owned to tenure. Designed to capture customer engagement over time and its impact on churn likelihood.
4. **Churn Probability Score**: A derived feature based on historical churn patterns, providing a probabilistic estimate of churn likelihood for each customer.

### Transformations:

1. **Scaling Numerical Features**: Applied `StandardScaler` to normalize features like `CreditScore`, `Age`, and `Balance`. This ensures that all numerical features are on a comparable scale, improving model convergence.
2. **Encoding Categorical Features**: Used one-hot encoding for variables like `Card Type` and `Geography` to convert them into numerical format. This transformation ensures that categorical variables are appropriately represented in the model.
3. **Log Transformation**: Applied to skewed features such as `Balance` and `EstimatedSalary` to reduce skewness and improve normality.
4. **Interaction Features**: Created new features by combining existing ones, such as `Age * Tenure` and `BalanceToSalary * ProductsPerTenure`, to capture complex relationships between variables.

### Expected Impact:

- Improved model performance by reducing feature skewness and enhancing interpretability.
- Better handling of categorical variables and numerical feature scaling to ensure consistency across the dataset.
- Enhanced predictive power through the addition of interaction features and derived metrics like `Churn Probability Score`.

---

## Visualization Summary

From the data visualization notebook:

- **Churn Rate by Tenure**: Boxplots revealed that customers with shorter tenures are more likely to churn. This insight underscores the importance of early engagement strategies.
- **Feature Importance**: Bar charts of logistic regression coefficients highlighted the top 10 features influencing churn, with `Tenure` and `BalanceToSalary` being the most significant.
- **Customer Segmentation**: Cluster plots based on `Balance` and `EstimatedSalary` identified distinct customer groups, aiding in targeted marketing strategies.

This report provides a comprehensive overview of the data analysis and feature engineering efforts, ensuring a robust foundation for predictive modeling.
