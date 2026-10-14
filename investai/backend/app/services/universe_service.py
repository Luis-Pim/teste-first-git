"""Universo de fundos (BRAPI + CVM), triagem de fundos novos e avaliação dos finalistas pelo motor."""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.domain.enums import AssetClass, InvestorProfile
from app.domain.screener import ScreenCriteria, UniverseFund, normalize_segment, screen
from app.market import cvm
from app.market.providers import MarketDataProvider, ProviderError, list_funds
from app.models import Asset, DataFetchLog, FundIndicator, FundUniverse, PriceQuote
from app.services import market_service, recommendation_service
from app.services import portfolio_service as svc


def _log(db: Session, provider: str, op: str) -> DataFetchLog:
    log = DataFetchLog(provider=provider, operation=op, status="RUNNING")
    db.add(log)
    db.flush()
    return log


def _finish(db: Session, log: DataFetchLog, status: str, detail: str) -> None:
    log.status, log.detail, log.finished_at = status, detail[:4000], datetime.now(timezone.utc)


def refresh_universe(db: Session, base_url: str, token: str | None, year: int, list_fetcher=None, cvm_fetcher=None) -> dict:
    """Atualiza a lista de fundos na BRAPI e enriquece com o informe mensal da CVM (o que estiver disponível)."""
    out = {"listed": 0, "with_cvm": 0, "warnings": []}
    log = _log(db, "BRAPI", "quote/list (fundos)")
    try:
        funds = (list_fetcher or (lambda: list_funds(base_url, token)))()
    except ProviderError as e:
        _finish(db, log, "ERROR", str(e))
        db.commit()
        out["warnings"].append(str(e))
        return out
    now = datetime.now(timezone.utc)
    for f in funds:
        db.execute(pg_insert(FundUniverse).values(ticker=f.ticker, name=f.name, sub_type=f.sub_type, close=f.close, volume=f.volume, listed_at=now)
                   .on_conflict_do_update(index_elements=["ticker"], set_={"name": f.name, "sub_type": f.sub_type, "close": f.close, "volume": f.volume, "listed_at": now}))
    out["listed"] = len(funds)
    _finish(db, log, "OK", f"fundos={len(funds)}")

    log = _log(db, "CVM", f"inf_mensal_fii_{year} (universo)")
    try:
        content, url = (cvm_fetcher or (lambda: cvm.fetch_zip(year)))()
        rows = cvm.parse_universe(content)
    except ProviderError as e:
        _finish(db, log, "ERROR", str(e))
        out["warnings"].append(f"CVM: {e}. A triagem segue sem segmento, patrimônio, cotistas e P/VP.")
        db.commit()
        return out
    by_root = {r.root: r for r in rows}
    enriched = 0
    for u in db.scalars(select(FundUniverse)):
        r = by_root.get(u.ticker[:4])
        if r is None:
            continue
        u.cnpj, u.segment, u.mandate, u.net_worth = r.cnpj, normalize_segment(r.segment), r.mandate, r.net_worth
        u.shareholders, u.vp_per_share, u.cvm_ref_date = r.shareholders, r.vp_per_share, r.ref_date
        if u.name is None or u.name == u.ticker:
            u.name = r.name
        enriched += 1
    out["with_cvm"] = enriched
    _finish(db, log, "OK", f"fundos_cvm={len(rows)} casados={enriched} url={url}")
    db.commit()
    return out


def run_screen(db: Session, pid: int, criteria: ScreenCriteria) -> dict:
    held = {p["ticker"] for p in svc.positions_view(db, pid, svc.today_br())["positions"] if p["quantity"] > 0}
    criteria.exclude |= held
    rows = list(db.scalars(select(FundUniverse)))
    universe = [UniverseFund(r.ticker, r.sub_type, r.close, r.volume, r.name, r.segment, r.mandate, r.net_worth, r.shareholders, r.vp_per_share)
                for r in rows]
    result = screen(universe, criteria)
    listed_at = max((r.listed_at for r in rows if r.listed_at), default=None)
    cvm_ref = max((r.cvm_ref_date for r in rows if r.cvm_ref_date), default=None)
    notes = ["A liquidez da lista vem do último pregão lido; a média real é calculada no histórico dos finalistas."]
    if not cvm_ref:
        notes.append("Sem dados da CVM no universo: segmento, patrimônio, cotistas e P/VP não foram verificados na triagem.")
    return {**result, "listed_at": listed_at, "cvm_ref_date": cvm_ref, "criteria": {
        "sub_types": sorted(criteria.sub_types), "segment": criteria.segment, "min_liquidity": criteria.min_liquidity,
        "min_net_worth": criteria.min_net_worth, "min_shareholders": criteria.min_shareholders,
        "pvp_range": [criteria.min_pvp, criteria.max_pvp]}, "notes": notes}


def evaluate_candidates(db: Session, pid: int, today: date, tickers: list[str], provider: MarketDataProvider,
                        profile: InvestorProfile | None = None, budget=None, cache_minutes: int = 15) -> dict:
    """Busca cotação, histórico de 12 meses e proventos dos finalistas e roda o motor com eles no universo.
    Candidatos entram como ativos inativos (não poluem a carteira nem a lista de acompanhamento)."""
    tickers = [t.upper() for t in tickers][:20]
    for t in tickers:
        u = db.get(FundUniverse, t)
        existed = db.scalar(select(Asset).where(Asset.ticker == t)) is not None
        a = svc.get_asset(db, t)
        if not existed:
            a.is_active = False  # candidato não entra na lista de acompanhamento sem pedido
        if a.name is None and u is not None:
            a.name, a.segment, a.cnpj = u.name, u.segment, u.cnpj
            a.asset_class = AssetClass.FII
            a.notes = "Candidato da triagem de fundos novos."
        if u is not None and u.close and u.listed_at:
            # cotação real da lista da BRAPI como ponto de partida; o provedor (com token) sobrescreve com dados completos
            db.execute(pg_insert(PriceQuote).values(asset_id=a.id, quote_date=u.listed_at.astimezone(svc.BR_TZ).date(), close=u.close,
                                                    volume=u.volume, source="BRAPI_LISTA")
                       .on_conflict_do_nothing(constraint="uq_quote_asset_date_source"))
        if u is not None and u.vp_per_share and u.cvm_ref_date:
            db.execute(pg_insert(FundIndicator).values(asset_id=a.id, ref_date=u.cvm_ref_date, vp_per_share=u.vp_per_share, source="CVM")
                       .on_conflict_do_nothing(constraint="uq_indicator_asset_date_source"))
    db.commit()
    fetch = market_service.refresh(db, provider, tickers, cache_minutes, with_dividends=True, history_range="1y")
    result = recommendation_service.recommendations(db, pid, today, profile, budget, extra_tickers=tickers)
    result["candidates"] = tickers
    result["fetch"] = {"quotes_saved": fetch["quotes_saved"], "dividends_saved": fetch["dividends_saved"], "missing": fetch["missing"]}
    if fetch["missing"]:
        result["notes"].append("Dados ausentes para alguns candidatos: " + "; ".join(f"{k}: {v}" for k, v in fetch["missing"].items()))
    return result


def set_watchlist(db: Session, ticker: str, active: bool) -> Asset:
    a = svc.get_asset(db, ticker)
    u = db.get(FundUniverse, a.ticker)
    if u is not None and a.name is None:
        a.name, a.segment, a.cnpj = u.name, u.segment, u.cnpj
    a.is_active = active
    db.commit()
    return a
