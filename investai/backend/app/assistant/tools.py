"""Ferramentas do assistente: única porta de acesso do modelo aos dados.

Cada ferramenta chama os mesmos serviços da API e devolve um resumo compacto. O modelo não tem acesso
a SQL nem à internet: o que não vier daqui, ele não sabe.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domain.enums import InvestorProfile
from app.domain.money import D
from app.domain.screener import ScreenCriteria
from app.domain.simulator import SCENARIOS, SimulationInput, extras_impact, required_contribution, simulate, time_to_income
from app.market.providers import get_provider
from app.services import portfolio_service as svc
from app.services import recommendation_service, universe_service

PROFILE_ENUM = ["CONSERVADOR", "EQUILIBRADO", "RENDA"]

TOOLS = [
    {"name": "get_portfolio", "description": "Carteira atual: posições (cotas, preço médio, cotação, valor de mercado, peso, segmento, DY 12m), totais, proventos recebidos e avisos de dados ausentes.", "parameters": {"type": "object", "properties": {}}},
    {"name": "get_budget", "description": "Orçamento de aporte: disponível no mês, aporte planejado, dividendos reinvestidos, saldo anterior e já utilizado; também a estratégia configurada.", "parameters": {"type": "object", "properties": {}}},
    {"name": "get_dividends", "description": "Proventos: recebidos, confirmados (anunciados, com data-com e cotas elegíveis), estimados por mês, e por fundo a soma/média de 12 meses, regularidade e tendência (queda ou alta dos últimos 3 meses).", "parameters": {"type": "object", "properties": {}}},
    {"name": "get_recommendations", "description": "Roda o motor de recomendação sobre a carteira e a lista de acompanhamento: sugestões com cotas, valor, justificativa, riscos e vetos de 'não comprar'.", "parameters": {"type": "object", "properties": {
        "profile": {"type": "string", "enum": PROFILE_ENUM, "description": "Perfil; omitido = perfil configurado"},
        "budget": {"type": "number", "description": "Valor a investir em R$; omitido = orçamento disponível do mês"}}}},
    {"name": "screen_new_funds", "description": "Triagem de FIIs que NÃO estão na carteira, a partir de todos os fundos listados: filtra por liquidez, patrimônio, cotistas, P/VP e segmento. Devolve a lista curta.", "parameters": {"type": "object", "properties": {
        "segment": {"type": "string", "description": "Trecho do segmento, ex.: 'logist', 'shopping', 'lajes', 'receb'"},
        "include_fiagro": {"type": "boolean"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 20}}}},
    {"name": "evaluate_new_funds", "description": "Busca dados de mercado de fundos candidatos (cotação, histórico, proventos) e roda o motor de recomendação com eles junto da carteira. Use depois de screen_new_funds.", "parameters": {"type": "object", "required": ["tickers"], "properties": {
        "tickers": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "profile": {"type": "string", "enum": PROFILE_ENUM}, "budget": {"type": "number"}}}},
    {"name": "simulate", "description": "Projeção patrimonial hipotética. Também responde 'em quanto tempo chego a R$ X/mês' (target_monthly_income) e 'quanto aportar para R$ X/mês em N anos' (solve_contribution=true).", "parameters": {"type": "object", "properties": {
        "monthly_contribution": {"type": "number"}, "years": {"type": "integer", "minimum": 1, "maximum": 30},
        "scenario": {"type": "string", "enum": ["conservador", "base", "otimista"]},
        "extras": {"type": "array", "items": {"type": "object", "properties": {"month": {"type": "integer"}, "amount": {"type": "number"}}}},
        "target_monthly_income": {"type": "number"}, "solve_contribution": {"type": "boolean"}}}},
    {"name": "add_to_watchlist", "description": "Adiciona um fundo à lista de acompanhamento (passa a entrar nas recomendações). Só use se o usuário pedir.", "parameters": {"type": "object", "required": ["ticker"], "properties": {"ticker": {"type": "string"}}}},
    {"name": "get_alerts", "description": "Alertas atuais: concentração, pagamentos próximos, pendências e dados ausentes.", "parameters": {"type": "object", "properties": {}}},
]


def _demo_flag(*sources) -> bool:
    return any(s and "DEMO" in str(s) for s in sources)


def t_get_portfolio(db: Session, pid: int, today: date, **_) -> dict:
    s = svc.summary(db, pid, today)
    pv = svc.positions_view(db, pid, today)
    return {
        "reference_date": today,
        "totals": {k: s[k] for k in ("market_value", "market_value_complete", "invested_total", "invested_pending", "unrealized_pnl",
                                     "dividends_month", "dividends_ytd", "dividends_12m", "dy_12m", "yoc_12m", "total_return")},
        "positions": [
            {"ticker": p["ticker"], "segment": p["segment"], "quantity": p["quantity"], "avg_price": p["avg_price"], "price": p["price"],
             "price_source": p["price_source"], "price_date": p["price_date"], "market_value": p["market_value"], "weight": p["weight"],
             "pending_quantity": p["pending_quantity"], "planned_quantity": p["planned_quantity"], "dy_12m": p["dividend_stats"]["dy_12m"]}
            for p in pv["positions"] if p["quantity"] > 0 or p["pending_quantity"] > 0 or p["planned_quantity"] > 0
        ],
        "price_sources": s["price_sources"],
        "data_gaps": s["notes"],
        "uses_demo_data": "DEMO" in s["price_sources"],
    }


def t_get_budget(db: Session, pid: int, today: date, **_) -> dict:
    lg = svc.ledger(db, pid, today, months_ahead=1)
    st = svc.get_strategy(db, pid, today)
    return {
        "reference_date": today, "current_month": lg["current"], "next_month": next((r for r in lg["rows"] if r["is_future"]), None),
        "strategy": {"monthly_contribution": st.monthly_contribution, "reinvest_pct": st.reinvest_pct, "annual_increase_pct": st.annual_increase_pct,
                     "extra_contributions": st.extra_contributions, "profile": st.profile,
                     "concentration_limit_asset": st.concentration_limit_asset, "concentration_limit_segment": st.concentration_limit_segment},
        "notes": lg["notes"],
    }


def t_get_dividends(db: Session, pid: int, today: date, **_) -> dict:
    inc = svc.income(db, pid, today, horizon_months=6)
    pv = svc.positions_view(db, pid, today)
    per_fund = {}
    for p in pv["positions"]:
        if p["quantity"] <= 0:
            continue
        st = svc.stats_for(db, pid, p["ticker"], today)
        per_fund[p["ticker"]] = {k: st[k] for k in ("sum_12m", "mean_monthly_12m", "regularity_12m", "coefficient_of_variation", "trend_last3_vs_prior", "dy_12m", "yoc_12m")}
        per_fund[p["ticker"]]["last"] = st["last"]
    return {
        "reference_date": today,
        "received_last": inc["received"][:12],
        "confirmed": [{k: c[k] for k in ("ticker", "status", "com_date", "payment_date", "amount_per_share", "eligible_quantity", "amount", "source")} for c in inc["confirmed"]],
        "estimated_by_month": inc["calendar"],
        "per_fund": per_fund,
        "notes": inc["notes"],
        "uses_demo_data": _demo_flag(*[c["source"] for c in inc["confirmed"]], *[v["last"]["source"] for v in per_fund.values() if v["last"]]),
    }


def _compact_rec(r: dict) -> dict:
    return {
        "reference_date": r["reference_date"], "profile": r["profile"], "budget": r["budget"], "budget_source": r["budget_source"],
        "allocated": r["allocated"], "remaining": r["remaining"],
        "picks": [{k: p[k] for k in ("ticker", "segment", "price", "quantity", "cost", "incremental_monthly_income", "weight_after", "score", "coverage", "justification", "risks")} for p in r["picks"]],
        "ranking": [{"ticker": e["ticker"], "score": e["score"], "coverage": e["coverage"], "eligible": e["eligible"], "vetoes": e["vetoes"],
                     "factors": {f["factor"]: {"score": f["score"], "detail": f["detail"], "source": f["source"]} for f in e["factors"]}} for e in r["ranking"]],
        "notes": r["notes"],
        "uses_demo_data": any("DEMO" in n for n in r["notes"]),
    }


def t_get_recommendations(db: Session, pid: int, today: date, profile: str | None = None, budget: float | None = None, **_) -> dict:
    r = recommendation_service.recommendations(db, pid, today, InvestorProfile(profile) if profile else None,
                                               D(str(budget)) if budget is not None else None)
    return _compact_rec(r)


def t_screen_new_funds(db: Session, pid: int, today: date, segment: str | None = None, include_fiagro: bool = False, max_results: int = 10, **_) -> dict:
    c = ScreenCriteria(sub_types={"fii"} | ({"fi-agro"} if include_fiagro else set()), segment=segment, max_results=max_results)
    r = universe_service.run_screen(db, pid, c)
    if r["universe_size"] == 0:
        r["notes"].insert(0, "Universo de fundos vazio: atualize em Configurações (Atualizar universo de fundos).")
    return r


def t_evaluate_new_funds(db: Session, pid: int, today: date, tickers: list[str], profile: str | None = None, budget: float | None = None, **_) -> dict:
    s = get_settings()
    r = universe_service.evaluate_candidates(db, pid, today, tickers[:10], get_provider(s), InvestorProfile(profile) if profile else None,
                                             D(str(budget)) if budget is not None else None, s.quote_cache_minutes)
    out = _compact_rec(r)
    out["candidates"], out["fetch"] = r["candidates"], r["fetch"]
    return out


def t_simulate(db: Session, pid: int, today: date, monthly_contribution: float = 300, years: int = 10, scenario: str = "base",
               extras: list | None = None, target_monthly_income: float | None = None, solve_contribution: bool = False, **_) -> dict:
    s = svc.summary(db, pid, today)
    initial = s["market_value"] if s["market_value_complete"] and s["market_value"] else s["invested_total"]
    inp = SimulationInput(initial_value=D(initial or 0), monthly_contribution=D(str(monthly_contribution)), years=max(1, min(30, years)),
                          extras=[(int(e["month"]), D(str(e["amount"]))) for e in (extras or [])],
                          start=date(today.year, today.month, 1), assumptions=SCENARIOS.get(scenario, SCENARIOS["base"]))
    out = {"assumptions": {k: str(v) for k, v in SCENARIOS.get(scenario, SCENARIOS["base"]).__dict__.items()}, "initial_value": inp.initial_value,
           "initial_value_source": "valor de mercado atual" if s["market_value_complete"] else "custo investido",
           "uses_demo_data": "DEMO" in s["price_sources"], "disclaimer": "Projeção hipotética; não é promessa de rentabilidade."}
    if target_monthly_income and solve_contribution:
        out["required_contribution"] = required_contribution(inp, D(str(target_monthly_income)), inp.years)
    elif target_monthly_income:
        out["time_to_target"] = time_to_income(inp, D(str(target_monthly_income)))
    else:
        out["final"] = simulate(inp, include_monthly=False)["final"]
        if extras:
            out["extras_impact"] = extras_impact(inp, inp.extras)
    return out


def t_add_to_watchlist(db: Session, pid: int, today: date, ticker: str, **_) -> dict:
    a = universe_service.set_watchlist(db, ticker, True)
    return {"ticker": a.ticker, "watchlist": True}


def t_get_alerts(db: Session, pid: int, today: date, **_) -> dict:
    return {"reference_date": today, "alerts": svc.alerts(db, pid, today)}


DISPATCH = {
    "get_portfolio": t_get_portfolio, "get_budget": t_get_budget, "get_dividends": t_get_dividends,
    "get_recommendations": t_get_recommendations, "screen_new_funds": t_screen_new_funds, "evaluate_new_funds": t_evaluate_new_funds,
    "simulate": t_simulate, "add_to_watchlist": t_add_to_watchlist, "get_alerts": t_get_alerts,
}


def to_jsonable(o):
    if isinstance(o, Decimal):
        return format(o.quantize(Decimal("1e-6")) if o.as_tuple().exponent < -6 else o, "f")
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if hasattr(o, "value") and not isinstance(o, (str, int, float, bool)):
        return o.value
    return o
