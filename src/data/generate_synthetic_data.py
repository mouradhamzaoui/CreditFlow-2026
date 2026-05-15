import os
import numpy as np
import pandas as pd


def generate_b2b_credit_data(n_samples=5000, random_state=42):
    """Génère un dataset synthétique réaliste de demandes de crédit B2B."""
    np.random.seed(random_state)

    # 1. Démographie de l'entreprise
    years_in_business = np.random.exponential(scale=8, size=n_samples).astype(int) + 1
    industry_sector = np.random.choice(
        ["Tech", "Retail", "Manufacturing", "Services", "Healthcare"],
        size=n_samples,
        p=[0.25, 0.20, 0.20, 0.25, 0.10],
    )

    # 2. Données Financières (Open Banking / ERP)
    annual_revenue = np.random.lognormal(mean=11.5, sigma=1.0, size=n_samples).round(
        2
    )  # En euros
    monthly_burn_rate = annual_revenue * np.random.uniform(
        0.05, 0.25, size=n_samples
    )  # Dépenses mensuelles
    debt_to_equity = np.random.uniform(0.1, 2.5, size=n_samples)

    # 3. Historique Bancaire
    previous_defaults = np.random.binomial(
        n=1, p=0.08, size=n_samples
    )  # 8% ont déjà fait un défaut
    avg_account_balance = annual_revenue * np.random.uniform(0.02, 0.15, size=n_samples)

    # 4. Données Modernes (ESG / Marché)
    esg_score = np.random.uniform(
        20, 100, size=n_samples
    )  # Score environnemental/social
    market_volatility = np.random.uniform(
        0.05, 0.40, size=n_samples
    )  # Volatilité du secteur

    # --- CALCUL DE LA CIBLE (Montant de Ligne de Crédit) ---
    # Formule réaliste simulée : basée sur les revenus, la santé financière, le risque et l'ESG
    base_credit = annual_revenue * 0.15  # Base : 15% du CA annuel

    # Ajustements
    health_multiplier = np.where(
        debt_to_equity > 1.5, 0.5, np.where(debt_to_equity < 0.5, 1.3, 1.0)
    )
    history_multiplier = np.where(previous_defaults == 1, 0.3, 1.0)
    esg_bonus = (esg_score / 100) * 0.15  # Jusqu'à 15% de bonus pour bon ESG
    risk_penalty = market_volatility * 0.2  # Pénalité pour volatilité

    # Montant final avec un peu de bruit aléatoire (pour simuler la réalité)
    credit_line_amount = (
        base_credit * health_multiplier * history_multiplier
        + (base_credit * esg_bonus)
        - (base_credit * risk_penalty)
        + np.random.normal(0, base_credit * 0.05, size=n_samples)  # Bruit
    ).round(0)

    # Le montant ne peut pas être négatif (minimum 0€)
    credit_line_amount = np.maximum(credit_line_amount, 0)

    # Création du DataFrame
    df = pd.DataFrame(
        {
            "YEARS_IN_BUSINESS": years_in_business,
            "INDUSTRY_SECTOR": industry_sector,
            "ANNUAL_REVENUE": annual_revenue,
            "MONTHLY_BURN_RATE": monthly_burn_rate.round(2),
            "DEBT_TO_EQUITY": debt_to_equity.round(2),
            "PREVIOUS_DEFAULTS": previous_defaults,
            "AVG_ACCOUNT_BALANCE": avg_account_balance.round(2),
            "ESG_SCORE": esg_score.round(1),
            "MARKET_VOLATILITY": market_volatility.round(2),
            "CREDIT_LINE_AMOUNT": credit_line_amount,  # CIBLE (Y)
        }
    )

    return df


def main():
    os.makedirs("data/raw", exist_ok=True)
    print("Génération du dataset synthétique B2B (5000 PME)...")
    df = generate_b2b_credit_data(n_samples=5000)

    file_path = "data/raw/b2b_credit_data.csv"
    df.to_csv(file_path, index=False)
    print(f"✅ Dataset sauvegardé dans {file_path}")
    print(f"Shape: {df.shape}")
    print("\nAperçu des 5 premières lignes :")
    print(df.head().to_string())
    print("\nStatistiques de la cible (CREDIT_LINE_AMOUNT) :")
    print(df["CREDIT_LINE_AMOUNT"].describe().to_string())


if __name__ == "__main__":
    main()
