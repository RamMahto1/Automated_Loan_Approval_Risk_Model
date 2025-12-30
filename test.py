import pandas as pd
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import sys

try:
    # Paths
    test_path = "artifacts/test.csv"
    preprocessor_path = "artifacts/preprocessor.pkl"
    model_path = "artifacts/best_model.pkl"

    # Load data, preprocessor, model
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    preprocessor = load_object(preprocessor_path)
    model = load_object(model_path)

    # Transform and predict
    X_test_trans = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_trans)

    # Evaluation
    logging.info(f"Accuracy: {accuracy_score(y_test,y_pred):.4f}, "
                 f"F1: {f1_score(y_test,y_pred):.4f}, "
                 f"Precision: {precision_score(y_test,y_pred):.4f}, "
                 f"Recall: {recall_score(y_test,y_pred):.4f}, "
                 f"ROC AUC: {roc_auc_score(y_test,y_pred):.4f}")

except Exception as e:
    raise CustomException(e, sys)
