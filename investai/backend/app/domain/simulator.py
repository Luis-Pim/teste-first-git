"""Simulador de crescimento patrimonial (projeção mensal, Decimal).

Modelo: a carteira é tratada como "unidades" a preço P (P0 = 1). A cada mês:
  1. P cresce pela valorização hipotética das cotas; o dividendo por unidade cresce pela taxa de
     crescimento de dividendos (ambas convertidas de anual para mensal composto);
  2. renda do mês = unidades × dividendo por unidade (yield anual inicial / 12);
  3. aporte do mês + dividendos reinvestidos compram novas unidades ao preço P.
Valores reais são deflacionados pela inflação hipotética acumulada.
Nada aqui é previsão: todas as hipóteses são devolvidas junto com o resultado.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date

from app.domain.dividends import add_months, month_key
from app.domain.money import ONE, ZERO, D, TWELVE, annual_to_monthly, money

MAX_YEARS = 30


@dataclass(frozen=True)
class Assumptions:
    dividend_yield_annual: object  # yield inicial sobre o patrimônio
    dividend_growth_annual: object
    price_growth_annual: object
    inflation_annual: object


SCENARIOS: dict[str, Assumptions] = {
    # Premissas ilustrativas e editáveis. Não são estimativas de mercado nem promessas.
    "conservador": Assumptions(D("0.085"), D("0.00"), D("0.00"), D("0.045")),
    "base": Assumptions(D("0.10"), D("0.03"), D("0.03"), D("0.04")),
    "otimista": Assumptions(D("0.11"), D("0.05"), D("0.05"), D("0.035")),
}


@dataclass
class SimulationInput:
    initial_value: object = ZERO
    monthly_contribution: object = D("300")
    annual_contribution_increase: object = ZERO
    extras: list[tuple[int, object]] = field(default_factory=list)  # (mês 1..12, valor)
    reinvest_pct: object = ONE
    years: int = 10
    start: date = date(2026, 1, 1)
    assumptions: Assumptions = SCENARIOS["base"]


def simulate(inp: SimulationInput, include_monthly: bool = True) -> dict:
    if not 1 <= inp.years <= MAX_YEARS:
        raise ValueError(f"Horizonte deve estar entre 1 e {MAX_YEARS} anos.")
    a = inp.assumptions
    gp = annual_to_monthly(a.price_growth_annual)
    gd = annual_to_monthly(a.dividend_growth_annual)
    gi = annual_to_monthly(a.inflation_annual)
    reinvest = D(inp.reinvest_pct)

    price = ONE
    div_per_unit = D(a.dividend_yield_annual) / TWELVE
    units = D(inp.initial_value)
    infl = ONE
    total_contrib = D(inp.initial_value)
    cum_reinvested = ZERO
    cum_dividends = ZERO
    last12: list = []
    monthly, yearly = [], []
    start = date(inp.start.year, inp.start.month, 1)

    for i in range(inp.years * 12):
        m = add_months(start, i)
        price *= ONE + gp
        div_per_unit *= ONE + gd
        infl *= ONE + gi
        income = units * div_per_unit
        reinvested = income * reinvest
        contrib = D(inp.monthly_contribution) * (ONE + D(inp.annual_contribution_increase)) ** (i // 12)
        contrib += sum((D(v) for mm, v in inp.extras if mm == m.month), ZERO)
        units += (contrib + reinvested) / price
        total_contrib += contrib
        cum_reinvested += reinvested
        cum_dividends += income
        last12 = (last12 + [income])[-12:]
        value = units * price
        row = {
            "month": month_key(m),
            "patrimonio": money(value),
            "patrimonio_real": money(value / infl),
            "total_aportado": money(total_contrib),
            "dividendos_reinvestidos_acum": money(cum_reinvested),
            "dividendos_acum": money(cum_dividends),
            "renda_mensal": money(income),
            "renda_mensal_real": money(income / infl),
            "renda_anual": money(sum(last12, ZERO)),
            "aporte_mes": money(contrib),
        }
        if include_monthly:
            monthly.append(row)
        if (i + 1) % 12 == 0:
            yearly.append({"year": (i + 1) // 12, **row})

    final = yearly[-1] if yearly else None
    return {
        "assumptions": {k: str(v) for k, v in asdict(a).items()},
        "inputs": {
            "initial_value": str(inp.initial_value),
            "monthly_contribution": str(inp.monthly_contribution),
            "annual_contribution_increase": str(inp.annual_contribution_increase),
            "extras": [{"month": mm, "amount": str(v)} for mm, v in inp.extras],
            "reinvest_pct": str(inp.reinvest_pct),
            "years": inp.years,
        },
        "monthly": monthly,
        "yearly": yearly,
        "final": final,
        "disclaimer": "Projeção hipotética baseada nas premissas informadas. Não constitui promessa de rentabilidade.",
    }


def scenarios(inp: SimulationInput, overrides: dict[str, Assumptions] | None = None) -> dict:
    out = {}
    for name, a in {**SCENARIOS, **(overrides or {})}.items():
        out[name] = simulate(replace(inp, assumptions=a), include_monthly=True)
    return out


def time_to_income(inp: SimulationInput, target_monthly, real: bool = False) -> dict:
    """Em quanto tempo a renda mensal atinge o alvo? Simula até 30 anos."""
    res = simulate(replace(inp, years=MAX_YEARS))
    key = "renda_mensal_real" if real else "renda_mensal"
    for idx, row in enumerate(res["monthly"]):
        if row[key] >= D(target_monthly):
            return {
                "reached": True,
                "months": idx + 1,
                "years": round((idx + 1) / 12, 1),
                "month": row["month"],
                "row": row,
                "assumptions": res["assumptions"],
            }
    return {"reached": False, "months": None, "best": res["final"], "assumptions": res["assumptions"]}


def required_contribution(inp: SimulationInput, target_monthly, years: int, real: bool = False) -> dict:
    """Aporte mensal necessário para atingir a renda alvo ao final do horizonte (busca binária)."""
    key = "renda_mensal_real" if real else "renda_mensal"

    def income_with(c) -> object:
        return simulate(replace(inp, monthly_contribution=c, years=years), include_monthly=False)["final"][key]

    if income_with(ZERO) >= D(target_monthly):
        return {"monthly_contribution": ZERO, "already_reached": True}
    lo, hi = ZERO, D("1000")
    while income_with(hi) < D(target_monthly):
        hi *= 2
        if hi > D("10000000"):
            return {"monthly_contribution": None, "already_reached": False, "feasible": False}
    while hi - lo > D("0.5"):
        mid = (lo + hi) / 2
        if income_with(mid) >= D(target_monthly):
            hi = mid
        else:
            lo = mid
    c = money(hi)
    final = simulate(replace(inp, monthly_contribution=c, years=years), include_monthly=False)
    return {"monthly_contribution": c, "already_reached": False, "feasible": True, "final": final["final"],
            "assumptions": final["assumptions"]}


def extras_impact(inp: SimulationInput, extras: list[tuple[int, object]]) -> dict:
    without = simulate(replace(inp, extras=[]), include_monthly=False)["final"]
    with_ = simulate(replace(inp, extras=extras), include_monthly=False)["final"]
    return {
        "without_extras": without,
        "with_extras": with_,
        "delta_patrimonio": money(with_["patrimonio"] - without["patrimonio"]),
        "delta_renda_mensal": money(with_["renda_mensal"] - without["renda_mensal"]),
        "delta_aportado": money(with_["total_aportado"] - without["total_aportado"]),
    }
