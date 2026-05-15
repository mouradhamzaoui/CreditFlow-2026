from pydantic import BaseModel, Field
from typing import List, Tuple


class CompanyData(BaseModel):
    """Données brutes de l'entreprise envoyées par le tableau de bord bancaire"""

    YEARS_IN_BUSINESS: int = Field(
        ...,
        description="Années d'existence de l'entreprise",
        json_schema_extra={"example": 5},
    )
    INDUSTRY_SECTOR: str = Field(
        ...,
        description="Secteur d'activité (Tech, Retail, Manufacturing, Services, Healthcare)",
        json_schema_extra={"example": "Tech"},
    )
    ANNUAL_REVENUE: float = Field(
        ...,
        description="Chiffre d'affaires annuel en €",
        json_schema_extra={"example": 150000.0},
    )
    MONTHLY_BURN_RATE: float = Field(
        ...,
        description="Dépenses mensuelles en €",
        json_schema_extra={"example": 12000.0},
    )
    DEBT_TO_EQUITY: float = Field(
        ...,
        description="Ratio Dette / Fonds propres",
        json_schema_extra={"example": 0.8},
    )
    PREVIOUS_DEFAULTS: int = Field(
        ...,
        description="Nombre de défauts passés (0 ou 1)",
        json_schema_extra={"example": 0},
    )
    AVG_ACCOUNT_BALANCE: float = Field(
        ...,
        description="Solde moyen du compte en €",
        json_schema_extra={"example": 25000.0},
    )
    ESG_SCORE: float = Field(
        ..., description="Score ESG (0-100)", json_schema_extra={"example": 75.0}
    )
    MARKET_VOLATILITY: float = Field(
        ...,
        description="Volatilité du marché (0.0-1.0)",
        json_schema_extra={"example": 0.15},
    )


class CreditDecisionResponse(BaseModel):
    """Réponse de l'API : Montant et Explications"""

    credit_line_amount_eur: float = Field(
        ..., description="Montant de la ligne de crédit recommandée en €"
    )
    confidence_score: float = Field(
        ..., description="Score de confiance du modèle (R2 simulé)"
    )
    risk_factors: List[Tuple[str, float]] = Field(
        ..., description="Top 3 des facteurs influençant la décision (SHAP)"
    )
    inference_time_ms: float = Field(
        ..., description="Temps de calcul en millisecondes"
    )
