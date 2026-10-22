"""
Regression Algorithms: Linear, Ridge (L2), Lasso (L1), and ElasticNet.
Demonstrates training, evaluation, and regularization comparison.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from utils.metrics import print_regression_summary


def compare_regression_models():
    print("Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "OLS Linear Regression": LinearRegression(),
        "Ridge Regression (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso Regression (alpha=0.01)": Lasso(alpha=0.01),
        "ElasticNet (alpha=0.01, l1_ratio=0.5)": ElasticNet(alpha=0.01, l1_ratio=0.5),
    }

    print("\nTraining and evaluating regression models:")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print_regression_summary(y_test, y_pred, title=name)


if __name__ == "__main__":
    compare_regression_models()
