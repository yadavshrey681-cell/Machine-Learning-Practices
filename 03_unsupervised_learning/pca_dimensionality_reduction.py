"""
Principal Component Analysis (PCA) Dimensionality Reduction.
Demonstrates feature compression, explained variance ratios, and 2D projections.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_pca_demonstration():
    print("Loading Iris dataset (4 dimensions)...")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA to check explained variance across all components
    pca_full = PCA()
    pca_full.fit(X_scaled)

    var_ratio = pca_full.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    print("\n--- Explained Variance Analysis ---")
    for i, (v, cv) in enumerate(zip(var_ratio, cum_var_ratio), 1):
        print(f"Component {i}: Variance Explained = {v * 100:6.2f}% | Cumulative = {cv * 100:6.2f}%")

    # Projecting to 2D
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)
    df_2d = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
    df_2d["Target"] = y.values

    print("\n--- 2D Projection Head ---")
    print(df_2d.head(6))
    print(f"\n2D projection retains {cum_var_ratio[1] * 100:.2f}% of the original 4-dimensional variance.")


if __name__ == "__main__":
    run_pca_demonstration()
