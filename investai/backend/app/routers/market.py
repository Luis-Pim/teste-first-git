from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.market.providers import get_provider
from app.models import Asset, DataFetchLog, DividendAnnouncement, PriceQuote
from app.routers.portfolio import PID, ref_date
from app.schemas import AnnouncementIn, AnnouncementOut, QuoteIn, QuoteOut
from app.services import market_service
from app.services import portfolio_service as svc

router = APIRouter(prefix="/api", tags=["mercado e proventos"])


# ------------------------------------------------------------------ cotações
@router.get("/quotes", response_model=list[QuoteOut])
def list_quotes(ticker: str | None = None, db: Session = Depends(get_db)):
    stmt = select(PriceQuote, Asset.ticker).join(Asset, Asset.id == PriceQuote.asset_id)
    if ticker:
        stmt = stmt.where(Asset.ticker == ticker.upper())
    rows = db.execute(stmt.order_by(PriceQuote.quote_date.desc()).limit(500))
    return [QuoteOut(ticker=t, quote_date=q.quote_date, close=q.close, source=q.source) for q, t in rows]


@router.post("/quotes", response_model=QuoteOut, status_code=201)
def upsert_quote(body: QuoteIn, db: Session = Depends(get_db)):
    a = svc.get_asset(db, body.ticker)
    stmt = pg_insert(PriceQuote).values(asset_id=a.id, quote_date=body.quote_date, close=body.close, source=body.source)
    db.execute(stmt.on_conflict_do_update(constraint="uq_quote_asset_date_source", set_={"close": body.close}))
    db.commit()
    return QuoteOut(ticker=a.ticker, quote_date=body.quote_date, close=body.close, source=body.source)


@router.post("/market/refresh")
def market_refresh(pid: int = PID, with_dividends: bool = False, db: Session = Depends(get_db)):
    s = get_settings()
    provider = get_provider(s)
    return market_service.refresh(db, provider, market_service.held_tickers(db, pid), s.quote_cache_minutes, with_dividends)


@router.get("/market/logs")
def market_logs(db: Session = Depends(get_db)):
    rows = db.scalars(select(DataFetchLog).order_by(DataFetchLog.id.desc()).limit(50))
    return [
        {"id": r.id, "provider": r.provider, "operation": r.operation, "status": r.status, "detail": r.detail,
         "started_at": r.started_at, "finished_at": r.finished_at}
        for r in rows
    ]


@router.get("/market/sources")
def market_sources():
    s = get_settings()
    return {
        "active_provider": s.market_provider,
        "brapi_token_configured": bool(s.brapi_token),
        "sources": [
            {"name": "Manual", "status": "ATIVO", "provides": ["cotações", "proventos"], "notes": "Cadastro pelo usuário; sempre disponível."},
            {"name": "BRAPI", "status": "ADAPTADOR PRONTO", "provides": ["cotação", "proventos (depende do plano)"],
             "notes": "Requer token. Plano gratuito tem limite mensal e pode não incluir proventos/histórico de FIIs."},
            {"name": "BCB SGS", "status": "ADAPTADOR PRONTO", "provides": ["CDI (12)", "Selic (11)", "IPCA (433)"], "notes": "Público, sem token."},
            {"name": "IBGE SIDRA", "status": "FASE 2", "provides": ["IPCA oficial"], "notes": "Alternativa ao SGS 433."},
            {"name": "CVM Dados Abertos", "status": "FASE 2", "provides": ["informe mensal FII: PL, VP/cota, cotistas"], "notes": "Arquivos CSV mensais."},
            {"name": "B3", "status": "FASE 2", "provides": ["IFIX", "eventos corporativos"], "notes": "Uso de dados sujeito aos termos da B3."},
            {"name": "Relatórios gerenciais", "status": "FASE 3", "provides": ["vacância", "inadimplência", "alavancagem"], "notes": "PDFs das gestoras → RAG."},
        ],
    }


# ------------------------------------------------------------------ proventos
def _ann_out(a: DividendAnnouncement) -> AnnouncementOut:
    return AnnouncementOut(
        id=a.id, ticker=a.asset.ticker, kind=a.kind, amount_per_share=a.amount_per_share, com_date=a.com_date,
        ex_date=a.ex_date, payment_date=a.payment_date, source=a.source, source_url=a.source_url,
    )


@router.get("/dividends/announcements", response_model=list[AnnouncementOut])
def list_announcements(ticker: str | None = None, db: Session = Depends(get_db)):
    stmt = select(DividendAnnouncement).join(Asset)
    if ticker:
        stmt = stmt.where(Asset.ticker == ticker.upper())
    return [_ann_out(a) for a in db.scalars(stmt.order_by(DividendAnnouncement.payment_date.desc()))]


@router.post("/dividends/announcements", response_model=AnnouncementOut, status_code=201)
def create_announcement(body: AnnouncementIn, db: Session = Depends(get_db)):
    a = svc.get_asset(db, body.ticker)
    data = body.model_dump(exclude={"ticker"})
    exists = db.scalar(
        select(DividendAnnouncement).where(
            DividendAnnouncement.asset_id == a.id, DividendAnnouncement.com_date == body.com_date,
            DividendAnnouncement.kind == body.kind, DividendAnnouncement.source == body.source,
        )
    )
    if exists:
        raise HTTPException(409, "Já existe anúncio para este ativo, data-com e tipo nesta fonte.")
    ann = DividendAnnouncement(asset_id=a.id, **data)
    db.add(ann)
    db.commit()
    return _ann_out(db.get(DividendAnnouncement, ann.id))


@router.delete("/dividends/announcements/{ann_id}", status_code=204)
def delete_announcement(ann_id: int, db: Session = Depends(get_db)):
    a = db.get(DividendAnnouncement, ann_id)
    if not a:
        raise HTTPException(404, "Anúncio não encontrado")
    db.delete(a)
    db.commit()


@router.get("/dividends/income")
def dividends_income(pid: int = PID, horizon_months: int = Query(12, ge=1, le=24), today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.income(db, pid, today, horizon_months)


@router.get("/dividends/stats/{ticker}")
def dividends_stats(ticker: str, pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.stats_for(db, pid, ticker.upper(), today)


@router.get("/assets/{ticker}/detail")
def asset_detail(ticker: str, pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    ticker = ticker.upper()
    a = svc.get_asset(db, ticker, create=False)
    if not a:
        raise HTTPException(404, "Ativo não encontrado")
    pos = next((p for p in svc.positions_view(db, pid, today)["positions"] if p["ticker"] == ticker), None)
    return {
        "asset": {"ticker": a.ticker, "name": a.name, "segment": a.segment, "asset_class": a.asset_class, "manager": a.manager, "cnpj": a.cnpj, "notes": a.notes},
        "position": pos,
        "dividend_stats": svc.stats_for(db, pid, ticker, today),
        "quotes": list_quotes(ticker, db)[:120],
        "indicators": {
            "available": False,
            "fields": ["P/VP", "vacância", "inadimplência", "alavancagem", "liquidez média diária"],
            "reason": "Indicadores fundamentalistas dependem das integrações CVM/relatórios (Fases 2 e 3).",
        },
    }
