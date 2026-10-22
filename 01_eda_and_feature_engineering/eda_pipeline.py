"""
Exploratory Data Analysis (EDA) Pipeline
Demonstrates statistical inspection, missing value profiling, distribution skewness,
and correlation mapping using scikit-learn datasets.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def generate_eda_report(df: pd.DataFrame, target_col: str = None) -> None:
    """Print a structured EDA summary of the dataset."""
    print("=" * 60)
    print(" DATASET EXPLORATION & STATISTICAL SUMMARY")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("[1] Column Types & Non-Null Counts:")
    dtype_summary = pd.DataFrame({
        "Data Type": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2),
    })
    print(dtype_summary)

    print("\n[2] Summary Statistics (Numerical):")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"}))

    print("\n[3] Skewness Analysis (|skew| > 1 indicates heavy skew):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[numeric_cols].skew().sort_values(ascending=False)
    for col, val in skewness.items():
        flag = " [High Skew]" if abs(val) > 1.0 else ""
        print(f"  - {col:20s}: {val:+.3f}{flag}")

    if target_col and target_col in df.columns:
        print(f"\n[4] Correlation with Target '{target_col}':")
        corr_with_target = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        print(corr_with_target.to_string())

    print("=" * 60)


def main():
    print("Loading California Housing dataset for EDA demonstration...")
    raw_data = fetch_california_housing(as_frame=True)
    df = raw_data.frame

    # Introduce some artificial missingness for demonstration
    np.random.seed(42)
    mask = np.random.rand(*df.shape) < 0.02
    df_with_missing = df.mask(mask)

    generate_eda_report(df_with_missing, target_col="MedHouseVal")


if __name__ == "__main__":
    main()
