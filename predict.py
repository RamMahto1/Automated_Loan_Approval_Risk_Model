import pandas as pd
import numpy as np
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
import sys

def predict(input_data, preprocessor_path, model_path):
    """
    Predict function for single row (dict) or batch (DataFrame)
    """
    try:
        # Load preprocessor and model
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)

        # Convert input dict to DataFrame if needed
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("input_data must be dict or pandas DataFrame")

        # Apply preprocessing
        X_transformed = preprocessor.transform(input_df)

        # Make prediction
        y_pred = model.predict(X_transformed)
        return y_pred

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    preprocessor_path = "artifacts/preprocessor.pkl"
    model_path = "artifacts/best_model.pkl"

    # Single row prediction
    sample_input = {
        "income": 8000,
        "age": 45,
        "loan": 14000
    }

    prediction = predict(sample_input, preprocessor_path, model_path)
    print(f"Prediction for sample input: {prediction}")

    # Batch prediction from CSV
    test_df = pd.read_csv("artifacts/test.csv")
    batch_pred = predict(test_df.drop(columns=["default"]), preprocessor_path, model_path)
    print(f"Batch predictions shape: {batch_pred.shape}")
