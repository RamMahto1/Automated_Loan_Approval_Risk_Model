import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Train models with GridSearchCV, evaluate, and return best model.
    Always returns 4 values.
    """
    try:
        best_score = -1
        best_model_name = None
        best_model = None
        report = []

        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")

            param_grid = params.get(model_name, {})

            # Use GridSearchCV if params are given
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1, error_score='raise')
                gs.fit(X_train, y_train)
                model = gs.best_estimator_  # replace model with best found
            else:
                model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics (use zero_division=0 to avoid errors)
            f1_train = f1_score(y_train, y_train_pred, zero_division=0)
            f1_test = f1_score(y_test, y_test_pred, zero_division=0)
            acc_test = accuracy_score(y_test, y_test_pred)
            precision_test = precision_score(y_test, y_test_pred, zero_division=0)
            recall_test = recall_score(y_test, y_test_pred, zero_division=0)
            roc_auc_test = roc_auc_score(y_test, y_test_pred)

            report.append({
                "Model": model_name,
                "Train F1": f1_train,
                "Test F1": f1_test,
                "Test Accuracy": acc_test,
                "Test Precision": precision_test,
                "Test Recall": recall_test,
                "Test ROC AUC": roc_auc_test
            })

            logging.info(f"{model_name} - Test F1: {f1_test:.4f}, Accuracy: {acc_test:.4f}")

            if f1_test > best_score:
                best_score = f1_test
                best_model_name = model_name
                best_model = model

        return report, best_model_name, best_score, best_model

    except Exception as e:
        logging.error(f"Error in evaluate_models: {e}")
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)  
    except Exception as e:
        raise CustomException(e, sys)