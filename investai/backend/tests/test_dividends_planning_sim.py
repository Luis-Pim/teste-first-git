from datetime import date
from decimal import Decimal as Dec

import pytest

from app.domain.dividends import AnnouncementInput, DividendTxn, dividend_stats, income_overview
from app.domain.enums import TxnStatus as S
from app.domain.enums import TxnType as T
from app.domain.planning import ExtraContribution, StrategyInput, affordable, budget_ledger, planned_contribution
from app.domain.portfolio import TxnInput
from app.domain.simulator import SCENARIOS, Assumptions, SimulationInput, extras_impact, required_contribution, simulate, time_to_income

TODAY = date(2026, 10, 11)


def buy(d, q, ticker="KNRI11", **kw):
    return TxnInput(type=T.BUY, trade_date=d, ticker=ticker, quantity=q, unit_price=Dec("100"), **kw)


# ---------- dividendos ----------

def test_confirmed_income_uses_eligible_quantity_on_com_date():
    txns = [buy(date(2026, 9, 1), 10), buy(date(2026, 10, 1), 5)]
    ann = [AnnouncementInput(id=1, ticker="KNRI11", amount_per_share=Dec("1.00"), com_date=date(2026, 9, 30), payment_date=date(2026, 10, 14))]
    out = income_overview(txns, [], ann, TODAY)
    c = out["confirmed"][0]
    assert c["eligible_quantity"] == 10 and c["amount"] == Dec("10.00") and c["status"] == "A_RECEBER"


def test_received_dividend_is_not_double_counted_as_confirmed():
    txns = [buy(date(2026, 8, 1), 10)]
    ann = [AnnouncementInput(id=1, ticker="KNRI11", amount_per_share=Dec("1"), com_date=date(2026, 8, 31), payment_date=date(2026, 9, 14))]
    rec = [DividendTxn(id=9, ticker="KNRI11", payment_date=date(2026, 9, 14), amount=Dec("10"), quantity=10)]
    out = income_overview(txns, rec, ann, TODAY)
    assert out["confirmed"] == []
    assert out["received"][0]["amount"] == Dec("10.00")


def test_future_com_date_marks_quantity_provisional():
    txns = [buy(date(2026, 9, 1), 10)]
    ann = [AnnouncementInput(id=1, ticker="KNRI11", amount_per_share=Dec("1"), com_date=date(2026, 10, 30), payment_date=date(2026, 11, 13))]
    c = income_overview(txns, [], ann, TODAY)["confirmed"][0]
    assert c["provisional_quantity"] is True


def test_estimated_income_excludes_months_already_confirmed_and_is_labelled():
    txns = [buy(date(2025, 1, 1), 10)]
    ann = [
        AnnouncementInput(id=i, ticker="KNRI11", amount_per_share=Dec("1.00"), com_date=date(2026, m, 1), payment_date=date(2026, m, 14))
        for i, m in enumerate(range(1, 11), start=1)
    ]
    out = income_overview(txns, [], ann, TODAY, horizon_months=3)
    months = [e["month"] for e in out["estimated"]]
    assert "2026-10" not in months  # já confirmado
    assert months == ["2026-11", "2026-12"]
    assert all(e["amount"] == Dec("10.00") for e in out["estimated"])
    assert "não é garantia" in out["notes"][0].lower() or "Não é garantia" in out["notes"][0]


def test_dividend_stats_yield_and_yoc():
    ann = [
        AnnouncementInput(id=m, ticker="KNRI11", amount_per_share=Dec("1.00"), com_date=date(2026, m, 1), payment_date=date(2026, m, 14))
        for m in range(1, 10)
    ] + [
        AnnouncementInput(id=100 + m, ticker="KNRI11", amount_per_share=Dec("1.00"), com_date=date(2025, m, 1), payment_date=date(2025, m, 14))
        for m in range(11, 13)
    ]
    s = dividend_stats("KNRI11", ann, [], [], TODAY, price=Dec("120"), avg_price=Dec("100"))
    assert s["sum_12m"] == Dec("11")
    assert s["regularity_12m"] == Dec("11") / 12
    assert s["dy_12m"] == Dec("11") / 120
    assert s["yoc_12m"] == Dec("0.11")
    assert s["stdev_12m"] == 0


# ---------- planejamento ----------

def test_budget_example_from_spec():
    # Aporte 300 + dividendos 85 + saldo anterior 40 = 425
    s = StrategyInput(monthly_contribution=Dec("300"), reinvest_pct=Dec("1"), start_month=date(2026, 10, 1), initial_cash=Dec("40"))
    txns = [TxnInput(type=T.DIVIDEND, trade_date=date(2026, 10, 14), ticker="KNRI11", total_amount=Dec("85"))]
    row = budget_ledger(s, txns, TODAY, months_ahead=0)[0]
    assert row["available"] == Dec("425.00") and row["own_source"] == "PLANEJADO"


def test_budget_residual_carries_to_next_month():
    s = StrategyInput(monthly_contribution=Dec("300"), start_month=date(2026, 9, 1))
    txns = [buy(date(2026, 9, 5), 2)]  # gasta 200
    rows = budget_ledger(s, txns, TODAY, months_ahead=0)
    assert rows[0]["carry_out"] == Dec("100.00")
    assert rows[1]["carry_in"] == Dec("100.00") and rows[1]["available"] == Dec("400.00")


def test_registered_deposit_overrides_plan_and_planned_orders_not_deducted():
    s = StrategyInput(monthly_contribution=Dec("300"), start_month=date(2026, 10, 1))
    txns = [
        TxnInput(type=T.CASH_DEPOSIT, trade_date=date(2026, 10, 2), total_amount=Dec("500")),
        buy(date(2026, 10, 20), 1, status=S.PLANNED),
    ]
    row = budget_ledger(s, txns, TODAY, months_ahead=0)[0]
    assert row["own_contribution"] == Dec("500.00") and row["own_source"] == "REGISTRADO"
    assert row["planned_orders"] == Dec("100.00") and row["carry_out"] == Dec("500.00")


def test_extras_and_annual_increase():
    s = StrategyInput(monthly_contribution=Dec("300"), annual_increase_pct=Dec("0.10"), start_month=date(2026, 1, 1),
                      extras=[ExtraContribution(month=6, amount=Dec("2000")), ExtraContribution(month=12, amount=Dec("3000"))])
    assert planned_contribution(s, date(2026, 6, 1)) == (Dec("300.00"), Dec("2000.00"))
    assert planned_contribution(s, date(2027, 1, 1)) == (Dec("330.00"), Dec("0.00"))


def test_affordable_quantities():
    out = affordable(Dec("425"), {"MXRF11": Dec("9.70"), "KNRI11": Dec("150")})
    by = {o["ticker"]: o for o in out}
    assert by["MXRF11"]["max_quantity"] == 43 and by["MXRF11"]["remaining"] == Dec("7.90")
    assert by["KNRI11"]["max_quantity"] == 2


# ---------- simulador ----------

ZERO_GROWTH = Assumptions(Dec("0.12"), Dec("0"), Dec("0"), Dec("0"))


def test_simulation_without_growth_is_deterministic():
    r = simulate(SimulationInput(initial_value=Dec("1000"), monthly_contribution=Dec("0"), reinvest_pct=Dec("0"), years=1, assumptions=ZERO_GROWTH))
    assert r["final"]["patrimonio"] == Dec("1000.00")
    assert r["final"]["renda_mensal"] == Dec("10.00")
    assert r["final"]["renda_anual"] == Dec("120.00")


def test_reinvestment_compounds():
    base = SimulationInput(initial_value=Dec("10000"), monthly_contribution=Dec("0"), years=5, assumptions=ZERO_GROWTH)
    with_r = simulate(base)["final"]["patrimonio"]
    without = simulate(SimulationInput(**{**base.__dict__, "reinvest_pct": Dec("0")}))["final"]["patrimonio"]
    assert with_r > without == Dec("10000.00")


def test_real_value_is_deflated():
    a = Assumptions(Dec("0.10"), Dec("0"), Dec("0"), Dec("0.05"))
    f = simulate(SimulationInput(initial_value=Dec("1000"), monthly_contribution=0, reinvest_pct=0, years=1, assumptions=a))["final"]
    assert f["patrimonio_real"] == Dec("952.38")


def test_goal_solvers():
    inp = SimulationInput(initial_value=Dec("5000"), monthly_contribution=Dec("300"), assumptions=SCENARIOS["base"])
    t = time_to_income(inp, Dec("500"))
    assert t["reached"] and 1 < t["years"] < 30
    rc = required_contribution(inp, Dec("2000"), years=15)
    assert rc["feasible"] and rc["final"]["renda_mensal"] >= Dec("2000")
    imp = extras_impact(inp, [(6, Dec("5000")), (12, Dec("5000"))])
    assert imp["delta_patrimonio"] > 0 and imp["delta_aportado"] == Dec("100000.00")


def test_horizon_validation():
    with pytest.raises(ValueError):
        simulate(SimulationInput(years=31))


def test_planned_order_without_price_valued_at_quote():
    s = StrategyInput(monthly_contribution=Dec("300"), start_month=date(2026, 10, 1))
    txns = [TxnInput(type=T.BUY, trade_date=date(2026, 10, 31), ticker="MXRF11", quantity=100, status=S.PLANNED)]
    row = budget_ledger(s, txns, TODAY, months_ahead=0, prices={"MXRF11": Dec("9.62")})[0]
    assert row["planned_orders"] == Dec("962.00") and row["carry_out"] == Dec("300.00")
