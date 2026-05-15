from src.api.schemas import CompanyData, CreditDecisionResponse


def test_company_data_schema_valid():
    data = {
        "YEARS_IN_BUSINESS": 5,
        "INDUSTRY_SECTOR": "Tech",
        "ANNUAL_REVENUE": 150000.0,
        "MONTHLY_BURN_RATE": 12000.0,
        "DEBT_TO_EQUITY": 0.8,
        "PREVIOUS_DEFAULTS": 0,
        "AVG_ACCOUNT_BALANCE": 25000.0,
        "ESG_SCORE": 75.0,
        "MARKET_VOLATILITY": 0.15,
    }
    company = CompanyData(**data)
    assert company.INDUSTRY_SECTOR == "Tech"


def test_credit_response_schema():
    response = CreditDecisionResponse(
        credit_line_amount_eur=25000.0,
        confidence_score=0.986,
        risk_factors=[("LOG_ANNUAL_REVENUE", 0.35)],
        inference_time_ms=120.5,
    )
    assert response.credit_line_amount_eur > 0
    assert response.inference_time_ms < 500  # KPI respecté
