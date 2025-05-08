import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create models directory if it doesn't exist
models_dir = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)

print(f"Saving models to: {models_dir}")

# Load data
data_path = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'data_after_preprocessing.csv')
data = pd.read_csv(data_path)

# Convert binary string columns to numeric
binary_columns = ['HasCrCard', 'IsActiveMember', 'Complain']
for col in binary_columns:
    data[col] = (data[col] == 1).astype(int)

# Separate features and target
target = 'Exited'
features = [col for col in data.columns if col != target]
X = data[features].copy()  # Create a copy to avoid SettingWithCopyWarning
y = data[target]

# Convert categorical variables
categorical_features = ['Card Type']
encoders = {}

for feature in categorical_features:
    # Create label encoder
    le = LabelEncoder()
    # Fit and transform the feature
    X.loc[:, feature] = le.fit_transform(X[feature])  # Use .loc to avoid SettingWithCopyWarning
    # Save the encoder
    encoder_path = os.path.join(models_dir, f'{feature.lower().replace(" ", "_")}_encoder.joblib')
    joblib.dump(le, encoder_path)
    encoders[feature] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
model_path = os.path.join(models_dir, 'churn_model.joblib')
joblib.dump(model, model_path)

# Save feature names
feature_names_path = os.path.join(models_dir, 'feature_names.joblib')
joblib.dump(list(X.columns), feature_names_path)

print("\nModel trained and saved successfully!")
print(f"Model saved to: {model_path}")
print(f"Feature names saved to: {feature_names_path}")
print("\nFeature names:", list(X.columns))
print("\nCard Type mapping:", dict(zip(encoders['Card Type'].classes_, encoders['Card Type'].transform(encoders['Card Type'].classes_)))) 