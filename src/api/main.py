import os
import time
import joblib
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from .schemas import CompanyData, CreditDecisionResponse

# --- CONFIGURATION DU LOGGING BANCAIRE ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Format JSON pour les audits de conformité
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "audit.log")),
        logging.StreamHandler(),
    ],
)
audit_logger = logging.getLogger("credit_audit")

# --- APPLICATION ---
app = FastAPI(
    title="CreditFlow 2026 API",
    description="API de prédiction de ligne de crédit B2B (Grade Bancaire).",
    version="2.0.0",
)

MODEL_DIR = "models"
ARTIFACTS_DIR = "src/models/artifacts"


@app.on_event("startup")
async def load_artifacts():
    global xgb_model, lgb_model, explainer, feature_names
    print("Chargement des artéfacts de production...")
    try:
        xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_regressor.joblib"))
        lgb_model = joblib.load(os.path.join(MODEL_DIR, "lgbm_regressor.joblib"))
        explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))
        feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))
        print("✅ Modèles chargés.")
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        raise e


def preprocess_input(data: CompanyData) -> pd.DataFrame:
    df = pd.DataFrame([data.dict()])
    log_cols = ["ANNUAL_REVENUE", "MONTHLY_BURN_RATE", "AVG_ACCOUNT_BALANCE"]
    for col in log_cols:
        df[f"LOG_{col}"] = np.log1p(df[col])
        df.drop(columns=[col], inplace=True)
    df = pd.get_dummies(
        df, columns=["INDUSTRY_SECTOR"], prefix="SECTOR", drop_first=True
    )
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df


@app.get("/health")
async def health_check():
    """Point de terminaison de santé pour le Cloud (Kubernetes/Load Balancer)"""
    return {"status": "UP", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/predict", response_model=CreditDecisionResponse)
async def predict_credit(company: CompanyData):
    start_time = time.time()
    request_id = f"REQ-{int(time.time()*1000)}"  # ID unique pour l'audit

    try:
        processed_df = preprocess_input(company)

        xgb_pred_log = xgb_model.predict(processed_df)[0]
        lgb_pred_log = lgb_model.predict(processed_df)[0]
        final_pred_log = (xgb_pred_log + lgb_pred_log) / 2

        credit_amount_eur = max(0, np.expm1(final_pred_log))

        shap_values = explainer.shap_values(processed_df)[0]
        feature_impact = list(zip(processed_df.columns, shap_values))
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        top_3 = [(feat, round(val, 4)) for feat, val in feature_impact[:3]]

        inference_time = (time.time() - start_time) * 1000

        # --- AUDIT LOGGING ---
        audit_log = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "company_sector": company.INDUSTRY_SECTOR,
            "company_revenue": company.ANNUAL_REVENUE,
            "predicted_credit_eur": round(credit_amount_eur, 2),
            "inference_ms": round(inference_time, 2),
            "top_risk_factor": top_3[0][0] if top_3 else None,
        }
        audit_logger.info(json.dumps(audit_log))

        return CreditDecisionResponse(
            credit_line_amount_eur=round(credit_amount_eur, 2),
            confidence_score=0.986,
            risk_factors=top_3,
            inference_time_ms=round(inference_time, 2),
        )

    except Exception as e:
        audit_logger.error(json.dumps({"request_id": request_id, "error": str(e)}))
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")
