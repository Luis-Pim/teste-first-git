from decimal import Decimal as Dec

from app.domain.enums import InvestorProfile as P
from app.domain.recommendation import FundMetrics, PortfolioContext, evaluate, recommend

CTX = PortfolioContext(
    values={"AAA11": Dec("1000"), "BBB11": Dec("1000"), "CCC11": Dec("1000"), "DDD11": Dec("1000")},
    segments={"AAA11": "Logística", "BBB11": "Shoppings", "CCC11": "Híbrido", "DDD11": "Recebíveis (CRI)"},
    limit_asset=Dec("0.25"),
    limit_segment=Dec("0.40"),
)


def fund(ticker="NEW11", segment="Lajes corporativas", price="100", **kw):
    base = dict(
        pvp=Dec("0.92"), dy_12m=Dec("0.095"), mean_monthly_per_share=Dec("0.80"), regularity_12m=Dec("1"),
        dividend_cv=Dec("0.05"), dividend_trend=Dec("0"), avg_daily_liquidity=Dec("3000000"),
        price_vs_12m_avg=Dec("0.97"), vacancy=Dec("0.05"), default_rate=Dec("0"), leverage=Dec("0.05"),
    )
    base.update(kw)
    return FundMetrics(ticker=ticker, segment=segment, price=Dec(price), **base)


def test_highest_yield_does_not_win_by_itself():
    good = fund("GOOD11")
    junk = fund("JUNK11", segment="Outros", dy_12m=Dec("0.17"), pvp=Dec("1.08"), regularity_12m=Dec("0.6"),
                dividend_cv=Dec("0.45"), dividend_trend=Dec("-0.10"), vacancy=Dec("0.18"), leverage=Dec("0.30"))
    r = recommend([good, junk], CTX, P.RENDA, budget=Dec("500"))
    assert r["picks"][0]["ticker"] == "GOOD11"
    junk_ev = next(e for e in r["ranking"] if e["ticker"] == "JUNK11")
    assert any("anormalmente alto" in x for x in junk_ev["risks"])


def test_dividend_cut_vetoes_purchase():
    ev = evaluate(fund(dividend_trend=Dec("-0.20")), CTX, P.EQUILIBRADO)
    assert not ev["eligible"] and any("Corte relevante" in v for v in ev["vetoes"])


def test_premium_over_book_value_vetoes_by_profile():
    f = fund(pvp=Dec("1.08"))
    assert not evaluate(f, CTX, P.CONSERVADOR)["eligible"]  # limite 1,05
    assert evaluate(f, CTX, P.EQUILIBRADO)["eligible"]  # limite 1,10


def test_low_liquidity_and_bad_fundamentals_veto():
    assert not evaluate(fund(avg_daily_liquidity=Dec("100000")), CTX, P.EQUILIBRADO)["eligible"]
    assert not evaluate(fund(vacancy=Dec("0.30")), CTX, P.EQUILIBRADO)["eligible"]
    assert not evaluate(fund(leverage=Dec("0.50")), CTX, P.EQUILIBRADO)["eligible"]
    # vacância não se aplica a fundo de papel
    assert evaluate(fund(vacancy=Dec("0.30"), is_paper=True, segment="Recebíveis (CRI)"), CTX, P.EQUILIBRADO)["eligible"]


def test_missing_data_redistributes_weight_and_reports_coverage():
    f = fund(vacancy=None, default_rate=None, leverage=None, price_vs_12m_avg=None)
    ev = evaluate(f, CTX, P.EQUILIBRADO)
    assert ev["eligible"] and ev["coverage"] == Dec("0.80")
    fund_f = next(x for x in ev["factors"] if x["factor"] == "fundamentals")
    assert fund_f["score"] is None and "relatório gerencial" in fund_f["detail"]
    assert any("não avaliados" in r for r in ev["risks"])


def test_insufficient_data_blocks_recommendation():
    f = FundMetrics(ticker="X11", segment="Logística", price=Dec("100"))
    ev = evaluate(f, CTX, P.EQUILIBRADO)
    assert not ev["eligible"] and any("Dados insuficientes" in v for v in ev["vetoes"])


def test_fund_already_above_limit_is_not_recommended():
    ctx = PortfolioContext({"BIG11": Dec("3000"), "AAA11": Dec("1000")}, {"BIG11": "Shoppings", "AAA11": "Logística"}, Dec("0.25"), Dec("0.80"))
    ev = evaluate(fund("BIG11", segment="Shoppings"), ctx, P.EQUILIBRADO)
    assert not ev["eligible"] and "Já representa 75,0%" in ev["vetoes"][0]


def test_allocation_respects_concentration_limit_and_spills_to_next():
    a = fund("NEW11", price="100")
    b = fund("OTH11", segment="Agro (Fiagro)", price="10", pvp=Dec("0.97"))
    r = recommend([a, b], CTX, P.EQUILIBRADO, budget=Dec("2000"))
    first = r["picks"][0]
    assert first["ticker"] == "NEW11"
    # 4000 de carteira: NEW11 pode ir até 25% do total após a compra → 13 cotas (1300/5300 = 24,5%)
    assert first["quantity"] == 13 and first["weight_after"] <= Dec("0.25")
    assert r["picks"][1]["ticker"] == "OTH11"
    assert r["allocated"] + r["remaining"] == Dec("2000.00")
    assert first["incremental_monthly_income"] == Dec("10.40")
    assert "Pontos fortes" in first["justification"]


def test_budget_smaller_than_any_price_buys_nothing():
    r = recommend([fund(price="150")], CTX, P.EQUILIBRADO, budget=Dec("100"))
    assert r["picks"] == [] and r["remaining"] == Dec("100.00")


def test_profiles_rank_differently():
    stable = fund("STB11", dy_12m=Dec("0.08"), dividend_cv=Dec("0.02"), avg_daily_liquidity=Dec("8000000"))
    yieldy = fund("YLD11", segment="Agro (Fiagro)", dy_12m=Dec("0.13"), dividend_cv=Dec("0.20"), avg_daily_liquidity=Dec("900000"))
    cons = recommend([stable, yieldy], CTX, P.CONSERVADOR, Dec("100"))
    renda = recommend([stable, yieldy], CTX, P.RENDA, Dec("100"))
    assert cons["ranking"][0]["ticker"] == "STB11"
    assert renda["ranking"][0]["ticker"] == "YLD11"
