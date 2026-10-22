"""
Classification Algorithms: Logistic Regression, Decision Tree, Support Vector Machine (SVM), and KNN.
Demonstrates training, evaluation, and confusion matrix calculation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from utils.metrics import print_classification_summary


def compare_classifiers():
    print("Loading Breast Cancer Wisconsin dataset (binary classification)...")
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
        "SVM (RBF Kernel)": SVC(probability=True, random_state=42),
        "k-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None
        print_classification_summary(y_test, y_pred, y_prob=y_prob, title=name)


if __name__ == "__main__":
    compare_classifiers()
