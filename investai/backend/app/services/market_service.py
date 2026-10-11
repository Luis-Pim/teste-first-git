"""Atualização de mercado com cache, controle de requisições e log de rastreabilidade."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.market.providers import MarketDataProvider
from app.models import Asset, DataFetchLog, DividendAnnouncement, PriceQuote, Transaction
from app.services.portfolio_service import get_asset


def held_tickers(db: Session, pid: int) -> list[str]:
    rows = db.execute(
        select(Asset.ticker).join(Transaction, Transaction.asset_id == Asset.id).where(Transaction.portfolio_id == pid).distinct()
    )
    return sorted(r[0] for r in rows)


def refresh(db: Session, provider: MarketDataProvider, tickers: list[str], cache_minutes: int, with_dividends: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    log = DataFetchLog(provider=provider.name, operation="quotes+dividends" if with_dividends else "quotes", status="RUNNING")
    db.add(log)
    db.flush()

    fresh_cut = now - timedelta(minutes=cache_minutes)
    cached = set(
        db.scalars(
            select(Asset.ticker)
            .join(PriceQuote, PriceQuote.asset_id == Asset.id)
            .where(PriceQuote.source == provider.name, PriceQuote.fetched_at >= fresh_cut, Asset.ticker.in_(tickers))
        )
    )
    to_fetch = [t for t in tickers if t not in cached or with_dividends]
    rep = provider.fetch(to_fetch, with_dividends=with_dividends) if to_fetch else None

    saved_q = saved_d = 0
    if rep:
        for q in rep.quotes:
            a = get_asset(db, q.ticker)
            stmt = pg_insert(PriceQuote).values(asset_id=a.id, quote_date=q.quote_date, close=q.close, source=q.source)
            stmt = stmt.on_conflict_do_update(constraint="uq_quote_asset_date_source", set_={"close": q.close, "fetched_at": now})
            db.execute(stmt)
            saved_q += 1
        for d in rep.dividends:
            a = get_asset(db, d.ticker)
            stmt = pg_insert(DividendAnnouncement).values(
                asset_id=a.id, kind=d.kind, amount_per_share=d.amount_per_share, com_date=d.com_date,
                payment_date=d.payment_date, source=d.source,
            ).on_conflict_do_nothing(constraint="uq_div_asset_com_kind_source").returning(DividendAnnouncement.id)
            saved_d += 1 if db.execute(stmt).scalar() is not None else 0

    missing = rep.missing if rep else {}
    log.status = "OK" if not missing else ("PARTIAL" if saved_q else "ERROR")
    log.detail = f"cache={sorted(cached)} fetched={to_fetch} quotes={saved_q} dividends={saved_d} missing={missing}"[:4000]
    log.finished_at = datetime.now(timezone.utc)
    db.commit()
    return {
        "provider": provider.name,
        "from_cache": sorted(cached),
        "quotes_saved": saved_q,
        "dividends_saved": saved_d,
        "missing": missing,
        "log_id": log.id,
    }
