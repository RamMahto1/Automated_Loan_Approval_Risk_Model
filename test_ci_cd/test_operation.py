# test_operation.py

import pytest
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -----------------------------
# Test preprocessing pipeline
# -----------------------------
def test_preprocessing_pipeline():
    # input with a missing value
    X = np.array([[50000, None, 10000]])
    
    # Imputer and scaler
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    
    assert X_scaled.shape == (1, 3)
    # All values must be finite numbers
    assert np.isfinite(X_scaled).all()

# -----------------------------
# Test model training and prediction
# -----------------------------
def test_model_prediction():
    # training data
    X_train = np.array([[50000, 30, 10000],
                        [60000, 40, 20000]])
    y_train = np.array([0, 1])
    
    # Train logistic regression
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    # Sample test data
    X_test = np.array([[55000, 35, 15000]])
    pred = model.predict(X_test)
    
    # Prediction checks
    assert pred.shape == (1,)
    # Should only predict 0 or 1
    assert pred[0] in [0, 1]

# -----------------------------
# Optional: Test multiple predictions
# -----------------------------
def test_batch_predictions():
    X_train = np.array([[50000, 30, 10000],
                        [60000, 40, 20000],
                        [70000, 50, 30000]])
    y_train = np.array([0, 1, 0])
    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    X_test = np.array([[52000, 33, 12000],
                       [65000, 45, 25000]])
    preds = model.predict(X_test)
    
    assert preds.shape == (2,)
    assert all([p in [0, 1] for p in preds])
