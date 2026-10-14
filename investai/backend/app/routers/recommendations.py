from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.domain.enums import InvestorProfile
from app.models import Asset, FundIndicator
from app.routers.portfolio import PID, ref_date
from app.services import portfolio_service as svc
from app.services import recommendation_service

router = APIRouter(prefix="/api", tags=["recomendações e indicadores"])


@router.get("/recommendations")
def get_recommendations(
    pid: int = PID,
    profile: InvestorProfile | None = None,
    budget: Decimal | None = Query(None, ge=0, description="Vazio = orçamento disponível do mês"),
    max_picks: int = Query(3, ge=1, le=7),
    today: date = Depends(ref_date),
    db: Session = Depends(get_db),
):
    return recommendation_service.recommendations(db, pid, today, profile, budget, max_picks)


class IndicatorIn(BaseModel):
    ticker: str
    ref_date: date
    vp_per_share: Decimal | None = Field(default=None, gt=0)
    vacancy: Decimal | None = Field(default=None, ge=0, le=1)
    default_rate: Decimal | None = Field(default=None, ge=0, le=1)
    leverage: Decimal | None = Field(default=None, ge=0, le=5)
    source: str = "MANUAL"
    source_url: str | None = None
    notes: str | None = None


def _out(i: FundIndicator) -> dict:
    return {
        "id": i.id, "ticker": i.asset.ticker, "ref_date": i.ref_date, "vp_per_share": i.vp_per_share,
        "vacancy": i.vacancy, "default_rate": i.default_rate, "leverage": i.leverage,
        "source": i.source, "source_url": i.source_url, "notes": i.notes,
    }


@router.get("/indicators")
def list_indicators(ticker: str | None = None, db: Session = Depends(get_db)):
    stmt = select(FundIndicator).join(Asset)
    if ticker:
        stmt = stmt.where(Asset.ticker == ticker.upper())
    return [_out(i) for i in db.scalars(stmt.order_by(FundIndicator.ref_date.desc()))]


@router.post("/indicators", status_code=201)
def upsert_indicator(body: IndicatorIn, db: Session = Depends(get_db)):
    if all(getattr(body, f) is None for f in ("vp_per_share", "vacancy", "default_rate", "leverage")):
        raise HTTPException(422, "Informe pelo menos um indicador.")
    a = svc.get_asset(db, body.ticker)
    ind = db.scalar(select(FundIndicator).where(
        FundIndicator.asset_id == a.id, FundIndicator.ref_date == body.ref_date, FundIndicator.source == body.source))
    if ind is None:
        ind = FundIndicator(asset_id=a.id, ref_date=body.ref_date, source=body.source)
        db.add(ind)
    for f in ("vp_per_share", "vacancy", "default_rate", "leverage", "source_url", "notes"):
        v = getattr(body, f)
        if v is not None:
            setattr(ind, f, v)
    db.commit()
    return _out(db.get(FundIndicator, ind.id))


@router.delete("/indicators/{ind_id}", status_code=204)
def delete_indicator(ind_id: int, db: Session = Depends(get_db)):
    i = db.get(FundIndicator, ind_id)
    if not i:
        raise HTTPException(404, "Indicador não encontrado")
    db.delete(i)
    db.commit()
