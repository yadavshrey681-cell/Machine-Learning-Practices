"""
Ensemble Learning: Bagging (Random Forest, Extra Trees) vs Boosting (Gradient Boosting).
Demonstrates ensemble training, performance comparison, and feature importance extraction.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from utils.metrics import print_classification_summary


def evaluate_ensembles():
    print("Loading Breast Cancer dataset for Ensemble comparison...")
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    ensembles = {
        "Random Forest (100 estimators)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Extra Trees Classifier (100 estimators)": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    feature_importances = {}

    for name, model in ensembles.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        print_classification_summary(y_test, y_pred, y_prob=y_prob, title=name)
        feature_importances[name] = model.feature_importances_

    # Feature Importance Comparison Table
    importance_df = pd.DataFrame(feature_importances, index=X.columns)
    print("=" * 60)
    print(" TOP 5 MOST IMPORTANT FEATURES (Random Forest vs Gradient Boosting)")
    print("=" * 60)
    top_rf = importance_df["Random Forest (100 estimators)"].sort_values(ascending=False).head(5)
    print("Random Forest Top Features:\n", top_rf)
    print("\nGradient Boosting Top Features:\n", importance_df["Gradient Boosting Classifier"].sort_values(ascending=False).head(5))
    print("=" * 60)


if __name__ == "__main__":
    evaluate_ensembles()
