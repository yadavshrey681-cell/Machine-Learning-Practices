"""
Feature Engineering Utilities
Provides modular recipes for missing value imputation, outlier handling via IQR,
and standard scaling pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer


def handle_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Cap numerical outliers using the Interquartile Range (IQR) rule."""
    df_cleaned = df.copy()
    for col in columns:
        q25 = df_cleaned[col].quantile(0.25)
        q75 = df_cleaned[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - (factor * iqr)
        upper_bound = q75 + (factor * iqr)

        num_capped = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
        df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"Column '{col}': Capped {num_capped} outlier values to [{lower_bound:.2f}, {upper_bound:.2f}]")
    return df_cleaned


def impute_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute numerical missing values using specified strategy (mean/median)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
    return df_imputed


def scale_features(df: pd.DataFrame, columns: list, method: str = "standard") -> tuple[pd.DataFrame, object]:
    """Scale specified columns with either StandardScaler or RobustScaler."""
    scaler = RobustScaler() if method == "robust" else StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled, scaler


def main():
    print("Testing Feature Engineering Utilities on sample synthetic data...")
    data = {
        "income": [25000, 32000, 48000, 52000, np.nan, 58000, 62000, 350000],  # Outlier: 350k
        "credit_score": [650, 710, np.nan, 780, 800, 720, 690, 850],
        "age": [22, 28, 35, 42, 50, 29, 31, 65],
    }
    df = pd.DataFrame(data)
    print("\n[Original Data with Missing and Outliers]:")
    print(df)

    print("\nStep 1: Imputing Missing Values (Median):")
    df_imputed = impute_missing_values(df, strategy="median")
    print(df_imputed)

    print("\nStep 2: Capping Outliers using IQR rule:")
    df_capped = handle_outliers_iqr(df_imputed, columns=["income"])

    print("\nStep 3: Scaling Features (StandardScaler):")
    df_scaled, _ = scale_features(df_capped, columns=["income", "credit_score", "age"])
    print(df_scaled.round(3))


if __name__ == "__main__":
    main()
