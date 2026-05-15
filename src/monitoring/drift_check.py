import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os


def check_data_drift():
    print("Vérification de la dérive des données (Data Drift)...")

    # Charger les données de référence (Train)
    reference_data = pd.read_parquet("data/processed/X_train.parquet")

    # Simuler de nouvelles données de production (ici on prend le Test pour l'exemple)
    production_data = pd.read_parquet("data/processed/X_test.parquet")

    # Créer le rapport Evidently
    report = Report(metrics=[DataDriftPreset()])

    report.run(reference_data=reference_data, current_data=production_data)

    # Sauvegarder le rapport HTML
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.html"
    report.save_html(report_path)

    # Extraire le résumé
    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    if drift_detected:
        print("⚠️ ALERTE : Dérive de données détectée ! Le modèle doit être réentraîné.")
    else:
        print("✅ Aucune dérive de données détectée. Le modèle est stable.")

    print(f"Rapport détaillé sauvegardé dans : {report_path}")


if __name__ == "__main__":
    check_data_drift()
