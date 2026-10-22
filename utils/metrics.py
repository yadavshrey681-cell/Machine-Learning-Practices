"""
Evaluation metric helper functions for Supervised Learning.
"""

from typing import Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
)


def print_classification_summary(y_true, y_pred, y_prob: Optional[np.ndarray] = None, title: str = "Classification Summary") -> None:
    """Print key classification metrics clearly in the console."""
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                print(f"ROC-AUC:   {roc_auc_score(y_true, prob):.4f}")
            else:
                print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob, multi_class='ovr'):.4f}")
        except Exception:
            pass

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"{'='*50}\n")


def print_regression_summary(y_true, y_pred, title: str = "Regression Summary") -> None:
    """Print key regression metrics (MAE, MSE, RMSE, R²)."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'='*20} {title} {'='*20}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"Mean Squared Error (MSE):      {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score (Coefficient of Det): {r2:.4f}")
    print(f"{'='*50}\n")
