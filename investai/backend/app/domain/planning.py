"""Planejamento de aportes: orçamento mensal = aporte próprio + dividendos reinvestidos + vendas + saldo anterior − compras."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.domain.dividends import add_months, month_key, months_between
from app.domain.enums import ACQUISITION_TYPES, TxnStatus, TxnType
from app.domain.money import ONE, ZERO, D, money
from app.domain.portfolio import TxnInput


@dataclass(frozen=True)
class ExtraContribution:
    month: int  # 1..12
    amount: object
    label: str = ""


@dataclass
class StrategyInput:
    monthly_contribution: object = D("300")
    reinvest_pct: object = D("1")
    annual_increase_pct: object = ZERO
    extras: list[ExtraContribution] = field(default_factory=list)
    start_month: date = date(2026, 1, 1)
    initial_cash: object = ZERO


def planned_contribution(s: StrategyInput, month: date) -> tuple[object, object]:
    """(aporte recorrente, aportes extraordinários) previstos para o mês."""
    k = max(0, months_between(s.start_month, month) // 12)
    base = money(D(s.monthly_contribution) * (ONE + D(s.annual_increase_pct)) ** k)
    extra = sum((D(e.amount) for e in s.extras if e.month == month.month), ZERO)
    return base, money(extra)


def _in_month(d: date, m: date) -> bool:
    return d.year == m.year and d.month == m.month


def _planned_value(t: TxnInput, prices: dict) -> object:
    """Valor da ordem planejada; sem preço informado, estima pela última cotação (rotulado na API)."""
    cost = t.gross_cost()
    if cost is not None:
        return cost
    price = prices.get(t.ticker)
    return D(t.quantity) * D(price) if price is not None and t.quantity is not None else ZERO


def budget_ledger(s: StrategyInput, txns: list[TxnInput], today: date, months_ahead: int = 1, prices: dict | None = None) -> list[dict]:
    prices = prices or {}
    start = date(s.start_month.year, s.start_month.month, 1)
    current = date(today.year, today.month, 1)
    end = add_months(max(current, start), months_ahead)
    rows: list[dict] = []
    carry = money(s.initial_cash)
    m = start
    while m <= end:
        in_m = [t for t in txns if _in_month(t.trade_date, m)]
        executed = [t for t in in_m if t.status in (TxnStatus.CONFIRMED, TxnStatus.PENDING)]
        deposits = sum((D(t.total_amount) for t in executed if t.type == TxnType.CASH_DEPOSIT), ZERO)
        withdrawals = sum((D(t.total_amount) for t in executed if t.type == TxnType.CASH_WITHDRAWAL), ZERO)
        dividends = sum(
            (D(t.total_amount) for t in executed if t.type == TxnType.DIVIDEND and t.status == TxnStatus.CONFIRMED), ZERO
        )
        amortizations = sum(
            (D(t.total_amount) for t in executed if t.type == TxnType.AMORTIZATION and t.status == TxnStatus.CONFIRMED),
            ZERO,
        )
        sales = sum(
            (t.sell_proceeds() for t in executed if t.type == TxnType.SELL and t.status == TxnStatus.CONFIRMED), ZERO
        )
        purchases = ZERO
        for t in executed:
            if t.type in ACQUISITION_TYPES and t.type != TxnType.OPENING_BALANCE:
                c = t.gross_cost()
                if c is not None:
                    purchases += c
        fees = sum((D(t.total_amount) for t in executed if t.type == TxnType.FEE), ZERO)
        planned_orders = sum(
            (_planned_value(t, prices) for t in in_m if t.status == TxnStatus.PLANNED and t.type in ACQUISITION_TYPES),
            ZERO,
        )

        planned_base, planned_extra = planned_contribution(s, m)
        has_actual = deposits > 0 or withdrawals > 0
        own = deposits - withdrawals if has_actual else planned_base + planned_extra
        reinvested = money(dividends * D(s.reinvest_pct))
        available = money(carry + own + reinvested + amortizations + sales)
        spent = money(purchases + fees)
        carry_out = money(available - spent)
        rows.append(
            {
                "month": month_key(m),
                "is_current": m == current,
                "is_future": m > current,
                "planned_contribution": planned_base,
                "planned_extra": planned_extra,
                "own_contribution": money(own),
                "own_source": "REGISTRADO" if has_actual else "PLANEJADO",
                "dividends_received": money(dividends),
                "dividends_reinvested": reinvested,
                "amortizations": money(amortizations),
                "sales": money(sales),
                "carry_in": carry,
                "available": available,
                "spent": spent,
                "planned_orders": money(planned_orders),
                "carry_out": carry_out,
            }
        )
        carry = carry_out
        m = add_months(m, 1)
    return rows


def affordable(available, prices: dict[str, object]) -> list[dict]:
    """Aritmética pura: quantas cotas de cada ativo cabem no orçamento (não é recomendação)."""
    out = []
    for ticker, price in sorted(prices.items()):
        if not price or D(price) <= 0:
            continue
        qty = int(D(available) // D(price))
        out.append(
            {
                "ticker": ticker,
                "price": D(price),
                "max_quantity": qty,
                "cost": money(qty * D(price)),
                "remaining": money(D(available) - qty * D(price)),
            }
        )
    return out
