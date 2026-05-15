import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

# Configuration des chemins
PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "src/models/artifacts"
MODEL_DIR = "models"


def load_data():
    print("Chargement des données traitées...")
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_val = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_val.parquet"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "y_train.parquet")
    ).values.ravel()
    y_val = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_val.parquet")).values.ravel()
    y_test = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "y_test.parquet")
    ).values.ravel()
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_base_models(X_train, y_train):
    print("Entraînement de XGBoost Regressor et LightGBM Regressor...")

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)

    return xgb_model, lgb_model


def ensemble_predict(models, X):
    """Moyenne des prédictions (en échelle Log) des deux modèles"""
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)


def calculate_metrics(y_true_log, y_pred_log):
    """Reconvertit en Euros réels et calcule les métriques financières"""
    y_true_eur = np.expm1(y_true_log)
    y_pred_eur = np.expm1(y_pred_log)

    # On s'assure qu'aucune prédiction n'est négative
    y_pred_eur = np.maximum(y_pred_eur, 0)

    mape = mean_absolute_percentage_error(y_true_eur, y_pred_eur) * 100  # En %
    r2 = r2_score(y_true_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))

    return mape, r2, rmse, y_true_eur, y_pred_eur


def generate_shap_explanations(model, X_test):
    print("Génération des explications SHAP...")
    explainer = shap.TreeExplainer(model)
    # Utiliser un sous-ensemble pour accélérer si le dataset est grand
    X_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(
        os.path.join(ARTIFACTS_DIR, "shap_summary_credit.png"), bbox_inches="tight"
    )
    plt.close()
    print("Graphique SHAP sauvegardé.")
    return explainer


def main():
    mlflow.set_experiment("CreditFlow_2026_Regression")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    with mlflow.start_run(run_name="XGB_LGBM_Ensemble_LogTarget_V1"):
        # 1. Entraînement
        xgb_model, lgb_model = train_base_models(X_train, y_train)
        models = [xgb_model, lgb_model]

        # 2. Évaluation sur Validation
        val_preds_log = ensemble_predict(models, X_val)
        val_mape, val_r2, val_rmse, _, _ = calculate_metrics(y_val, val_preds_log)

        print("\n--- [VAL] Résultats sur VALIDATION ---")
        print(f"MAPE: {val_mape:.2f}%")
        print(f"R²:   {val_r2:.4f}")
        print(f"RMSE: {val_rmse:.2f} €")

        # 3. Évaluation finale sur TEST (les données jamais vues)
        test_preds_log = ensemble_predict(models, X_test)
        test_mape, test_r2, test_rmse, y_test_eur, y_pred_eur = calculate_metrics(
            y_test, test_preds_log
        )

        print("\n--- [TEST] Résultats sur TEST (Objectifs) ---")
        print(f"MAPE: {test_mape:.2f}% (Objectif < 8%)")
        print(f"R²:   {test_r2:.4f} (Objectif > 0.85)")
        print(f"RMSE: {test_rmse:.2f} €")

        # Exemple de prédictions réelles
        print("\nExemples de prédictions (vs Réalité) :")
        for i in range(5):
            print(
                f"  Réel: {y_test_eur[i]:>10.2f} € | Prédit: {y_pred_eur[i]:>10.2f} € | Erreur: {abs(y_test_eur[i] - y_pred_eur[i]):>8.2f} €"
            )

        # 4. Log MLflow
        mlflow.log_params(
            {
                "model_type": "XGB + LGBM Ensemble",
                "n_estimators": 300,
                "target_transform": "log1p",
            }
        )
        mlflow.log_metrics(
            {"test_mape_pct": test_mape, "test_r2": test_r2, "test_rmse_eur": test_rmse}
        )

        # 5. Explicabilité SHAP
        explainer = generate_shap_explanations(xgb_model, X_test)
        mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "shap_summary_credit.png"))

        # 6. Sauvegarde des artéfacts de production
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_regressor.joblib"))
        joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lgbm_regressor.joblib"))
        joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.joblib"))

        mlflow.log_artifact(os.path.join(MODEL_DIR, "xgb_regressor.joblib"))
        mlflow.log_artifact(os.path.join(MODEL_DIR, "lgbm_regressor.joblib"))

    print("\n[OK] Phase 3 terminee : Modeles de Regression entraines et evalues !")


if __name__ == "__main__":
    main()
