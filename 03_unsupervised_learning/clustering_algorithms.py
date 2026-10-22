"""
Clustering Algorithms: K-Means (Elbow Method & Silhouette Score) and DBSCAN.
Demonstrates unsupervised grouping on synthetic and standard benchmark data.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def evaluate_kmeans_and_dbscan():
    print("Generating synthetic clustering dataset (4 centers, 600 samples)...")
    X, y_true = make_blobs(n_samples=600, centers=4, cluster_std=0.85, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- [1] K-Means Clustering: Elbow Analysis & Silhouette Scores ---")
    best_k = None
    best_silhouette = -1

    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, cluster_labels)
        print(f"k = {k:2d} | Inertia: {kmeans.inertia_:10.2f} | Silhouette Score: {sil:.4f}")

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    print(f"\nOptimal k according to Silhouette Score: k = {best_k} (Score: {best_silhouette:.4f})")

    print("\n--- [2] Density-Based Spatial Clustering (DBSCAN) ---")
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    db_labels = dbscan.fit_predict(X_scaled)

    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    print(f"DBSCAN detected {n_clusters_db} clusters and {n_noise} noise points (outliers).")

    if n_clusters_db > 1:
        valid_mask = db_labels != -1
        if valid_mask.sum() > 0:
            sil_db = silhouette_score(X_scaled[valid_mask], db_labels[valid_mask])
            print(f"DBSCAN Silhouette Score (excluding noise): {sil_db:.4f}")


if __name__ == "__main__":
    evaluate_kmeans_and_dbscan()
