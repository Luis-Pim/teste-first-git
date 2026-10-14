"""Monta as métricas de cada fundo a partir do banco e chama o motor de recomendação."""

from __future__ import annotations

from datetime import date, timedelta
from statistics import median

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.dividends import add_months
from app.domain.enums import InvestorProfile
from app.domain.money import ZERO, D, money
from app.domain.recommendation import FundMetrics, PortfolioContext, recommend
from app.models import Asset, FundIndicator, PriceQuote
from app.services import portfolio_service as svc

LIQUIDITY_WINDOW_DAYS = 30
INDICATOR_MAX_AGE_DAYS = 200  # indicador mais velho que isso é ignorado (e reportado)


def _quotes_by_ticker(db: Session, since: date) -> dict[str, list[PriceQuote]]:
    rows = db.execute(
        select(PriceQuote, Asset.ticker).join(Asset, Asset.id == PriceQuote.asset_id)
        .where(PriceQuote.quote_date >= since).order_by(PriceQuote.quote_date)
    )
    out: dict[str, list[PriceQuote]] = {}
    for q, t in rows:
        out.setdefault(t, []).append(q)
    return out


def _indicators_by_ticker(db: Session) -> dict[str, list[FundIndicator]]:
    out: dict[str, list[FundIndicator]] = {}
    for ind in db.scalars(select(FundIndicator).order_by(FundIndicator.ref_date)):
        out.setdefault(ind.asset.ticker, []).append(ind)
    return out


def _fmt(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def build_metrics(db: Session, pid: int, today: date, extra_tickers: list[str] | None = None) -> tuple[list[FundMetrics], PortfolioContext, dict]:
    pv = svc.positions_view(db, pid, today)
    positions = {p["ticker"]: p for p in pv["positions"]}
    quotes12 = _quotes_by_ticker(db, add_months(today, -12))
    latest = svc.latest_quotes(db)
    indicators = _indicators_by_ticker(db)
    strategy = svc.get_strategy(db, pid, today)
    cond = Asset.is_active.is_(True)
    if extra_tickers:
        cond = cond | Asset.ticker.in_(extra_tickers)
    assets = list(db.scalars(select(Asset).where(cond).order_by(Asset.ticker)))
    data_notes: list[str] = []
    demo_sources: set[str] = set()

    metrics: list[FundMetrics] = []
    for a in assets:
        t = a.ticker
        q = latest.get(t)
        price = D(q.price) if q else None
        stats = svc.stats_for(db, pid, t, today)
        sources: dict[str, str] = {}
        if q:
            sources["price"] = f"{q.source} {_fmt(q.quote_date)}"
            if q.source == "DEMO":
                demo_sources.add("cotações")

        # Liquidez: média de (cotas negociadas × preço) nos últimos 30 dias com volume informado.
        liq = None
        recent = [x for x in quotes12.get(t, []) if x.volume is not None and x.quote_date >= today - timedelta(days=LIQUIDITY_WINDOW_DAYS)]
        if recent:
            liq = sum((D(x.volume) * D(x.close) for x in recent), ZERO) / len(recent)
            sources["liquidity"] = f"{recent[-1].source}, média de {len(recent)} pregão(ões)"

        # Preço vs. média de 12 meses: só com histórico de pelo menos 6 meses.
        pos12 = None
        hist = quotes12.get(t, [])
        if price and hist and (hist[-1].quote_date - hist[0].quote_date).days >= 180:
            avg = sum((D(x.close) for x in hist), ZERO) / len(hist)
            pos12 = price / avg
            sources["price_position"] = f"{hist[-1].source}, {len(hist)} cotações em 12 meses"

        # Fundamentos: último indicador com cada campo, dentro da validade.
        inds = [i for i in indicators.get(t, []) if (today - i.ref_date).days <= INDICATOR_MAX_AGE_DAYS]
        stale = [i for i in indicators.get(t, []) if (today - i.ref_date).days > INDICATOR_MAX_AGE_DAYS]
        if stale and not inds:
            data_notes.append(f"{t}: indicadores mais antigos que {INDICATOR_MAX_AGE_DAYS} dias foram ignorados.")

        def last(field):
            for i in reversed(inds):
                v = getattr(i, field)
                if v is not None:
                    return i, D(v)
            return None, None

        vp_ind, vp = last("vp_per_share")
        pvp = price / vp if (price and vp) else None
        pvp_med = None
        if pvp is not None:
            sources["valuation"] = f"{vp_ind.source} ref. {_fmt(vp_ind.ref_date)}"
            if vp_ind.source == "DEMO":
                demo_sources.add("indicadores")
            hist_pvp = []
            for i in inds + [x for x in indicators.get(t, []) if x not in inds and (today - x.ref_date).days <= 365]:
                if i.vp_per_share is None:
                    continue
                qs = [x for x in hist if x.quote_date <= i.ref_date]
                if qs:
                    hist_pvp.append(D(qs[-1].close) / D(i.vp_per_share))
            if len(hist_pvp) >= 3:
                pvp_med = D(median(hist_pvp))

        fund_parts = []
        vac_i, vacancy = last("vacancy")
        def_i, default = last("default_rate")
        lev_i, leverage = last("leverage")
        for label, ind in (("vacância", vac_i), ("inadimplência", def_i), ("alavancagem", lev_i)):
            if ind:
                fund_parts.append(f"{label}: {ind.source} {_fmt(ind.ref_date)}")
                if ind.source == "DEMO":
                    demo_sources.add("indicadores")
        if fund_parts:
            sources["fundamentals"] = "; ".join(fund_parts)

        if stats["data_points"]:
            last_src = stats["last"]["source"] if stats["last"] else "registros"
            sources["income"] = sources["income_quality"] = f"proventos ({last_src}), {stats['data_points']} registro(s)"
            if last_src == "DEMO":
                demo_sources.add("proventos")

        p = positions.get(t)
        metrics.append(FundMetrics(
            ticker=t, segment=a.segment, price=price,
            held_quantity=D(p["quantity"]) if p else ZERO,
            pvp=pvp, pvp_median_12m=pvp_med,
            dy_12m=stats["dy_12m"], mean_monthly_per_share=stats["mean_monthly_12m"],
            regularity_12m=stats["regularity_12m"] if stats["data_points"] else None,
            dividend_cv=stats["coefficient_of_variation"], dividend_trend=stats["trend_last3_vs_prior"],
            avg_daily_liquidity=liq, price_vs_12m_avg=pos12,
            vacancy=vacancy, default_rate=default, leverage=leverage,
            is_paper=bool(a.segment and "receb" in a.segment.lower()),
            sources=sources,
        ))

    values = {t: D(p["market_value"]) for t, p in positions.items() if p["quantity"] > 0 and p["market_value"] is not None}
    missing_value = [t for t, p in positions.items() if p["quantity"] > 0 and p["market_value"] is None]
    if missing_value:
        data_notes.append(f"Sem cotação para {', '.join(missing_value)}: esses fundos ficam fora do cálculo de concentração.")
    ctx = PortfolioContext(
        values=values,
        segments={a.ticker: a.segment for a in assets},
        limit_asset=D(strategy.concentration_limit_asset),
        limit_segment=D(strategy.concentration_limit_segment),
    )
    meta = {"notes": data_notes, "demo_sources": sorted(demo_sources), "profile_default": strategy.profile}
    return metrics, ctx, meta


def recommendations(db: Session, pid: int, today: date, profile: InvestorProfile | None, budget=None, max_picks: int = 3,
                    extra_tickers: list[str] | None = None) -> dict:
    metrics, ctx, meta = build_metrics(db, pid, today, extra_tickers)
    profile = profile or InvestorProfile(meta["profile_default"])
    budget_source = "informado"
    if budget is None:
        current = svc.ledger(db, pid, today, months_ahead=0)["current"]
        budget = D(current["available"]) if current else ZERO
        budget_source = f"orçamento disponível em {current['month']}" if current else "sem orçamento"
    result = recommend(metrics, ctx, profile, money(budget), max_picks=max_picks)
    notes = list(meta["notes"])
    if meta["demo_sources"]:
        notes.insert(0, f"Atenção: {', '.join(meta['demo_sources'])} usam dados DEMO (ilustrativos). O resultado serve só para testar o motor.")
    notes.append("Renda incremental estimada = média mensal por cota dos últimos 12 meses × cotas sugeridas. Não é garantia.")
    return {
        **result,
        "reference_date": today,
        "budget_source": budget_source,
        "universe": [m.ticker for m in metrics],
        "notes": notes,
        "disclaimer": "Apoio à decisão baseado nos dados registrados. Não é recomendação de investimento nem promessa de rentabilidade.",
    }
