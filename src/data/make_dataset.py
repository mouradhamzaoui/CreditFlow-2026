import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Configuration des chemins
RAW_DATA_PATH = "data/raw/b2b_credit_data.csv"
PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "src/models/artifacts"


def load_data(path):
    print(f"Chargement des données depuis {path}...")
    return pd.read_csv(path)


def preprocess_features(df):
    """Application des transformations financières standard (Log & Encoding)"""
    print("Prétraitement des features...")

    # 1. Log Transformation des variables financières skewées (pour aider le modèle)
    log_transform_cols = ["ANNUAL_REVENUE", "MONTHLY_BURN_RATE", "AVG_ACCOUNT_BALANCE"]
    for col in log_transform_cols:
        df[f"LOG_{col}"] = np.log1p(df[col])
        df.drop(columns=[col], inplace=True)

    # 2. One-Hot Encoding de la variable catégorielle (INDUSTRY_SECTOR)
    df = pd.get_dummies(
        df, columns=["INDUSTRY_SECTOR"], prefix="SECTOR", drop_first=True
    )

    return df


def preprocess_target(y):
    """Log Transformation de la cible (Montant du crédit)"""
    print("Application du Log1p sur la cible (CREDIT_LINE_AMOUNT)...")
    return np.log1p(y)


def main():
    # 1. Chargement
    df = load_data(RAW_DATA_PATH)

    # 2. Séparation Features / Cible
    y = df["CREDIT_LINE_AMOUNT"]
    X = df.drop(columns=["CREDIT_LINE_AMOUNT"])

    # 3. Transformation des Features
    X = preprocess_features(X)

    # 4. Transformation de la Cible
    y_log = preprocess_target(y)

    # 5. Sauvegarde des noms de colonnes (CRUCIAL pour l'API de production plus tard)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))
    print(
        f"Artéfact 'feature_names.joblib' sauvegardé ({len(feature_names)} features)."
    )

    # 6. Split (70% Train, 15% Val, 15% Test)
    print("Découpage Train / Val / Test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_log, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(
        f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}"
    )

    # 7. Sauvegarde au format Parquet
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X_train.to_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"), index=False)
    X_val.to_parquet(os.path.join(PROCESSED_DIR, "X_val.parquet"), index=False)
    X_test.to_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"), index=False)

    pd.DataFrame(y_train).to_parquet(
        os.path.join(PROCESSED_DIR, "y_train.parquet"), index=False
    )
    pd.DataFrame(y_val).to_parquet(
        os.path.join(PROCESSED_DIR, "y_val.parquet"), index=False
    )
    pd.DataFrame(y_test).to_parquet(
        os.path.join(PROCESSED_DIR, "y_test.parquet"), index=False
    )

    print("\n✅ Pipeline de données terminée ! (Cible transformée en Log)")


if __name__ == "__main__":
    main()
