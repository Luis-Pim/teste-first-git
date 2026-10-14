from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.domain.enums import InvestorProfile
from app.domain.screener import ScreenCriteria
from app.market.providers import get_provider
from app.routers.portfolio import PID, ref_date
from app.services import portfolio_service as svc
from app.services import universe_service

router = APIRouter(prefix="/api", tags=["fundos novos"])


@router.post("/universe/refresh")
def refresh_universe(year: int | None = None, db: Session = Depends(get_db)):
    s = get_settings()
    return universe_service.refresh_universe(db, s.brapi_base_url, s.brapi_token, year or svc.today_br().year)


@router.get("/universe/screen")
def screen_funds(
    pid: int = PID,
    segment: str | None = None,
    include_fiagro: bool = False,
    include_fiinfra: bool = False,
    min_liquidity: Decimal = Query(Decimal("300000"), ge=0),
    min_net_worth: Decimal = Query(Decimal("300000000"), ge=0),
    min_shareholders: int = Query(10000, ge=0),
    max_pvp: Decimal = Query(Decimal("1.10"), gt=0),
    require_cvm: bool = False,
    max_results: int = Query(15, ge=1, le=50),
    db: Session = Depends(get_db),
):
    types = {"fii"} | ({"fi-agro"} if include_fiagro else set()) | ({"fi-infra"} if include_fiinfra else set())
    c = ScreenCriteria(sub_types=types, segment=segment, min_liquidity=min_liquidity, min_net_worth=min_net_worth,
                       min_shareholders=min_shareholders, max_pvp=max_pvp, require_cvm=require_cvm, max_results=max_results)
    return universe_service.run_screen(db, pid, c)


class EvaluateIn(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=20)
    profile: InvestorProfile | None = None
    budget: Decimal | None = Field(default=None, ge=0)


@router.post("/universe/evaluate")
def evaluate(body: EvaluateIn, pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    s = get_settings()
    return universe_service.evaluate_candidates(db, pid, today, body.tickers, get_provider(s), body.profile, body.budget, s.quote_cache_minutes)


@router.post("/watchlist/{ticker}")
def add_watchlist(ticker: str, db: Session = Depends(get_db)):
    a = universe_service.set_watchlist(db, ticker, True)
    return {"ticker": a.ticker, "watchlist": True}


@router.delete("/watchlist/{ticker}")
def remove_watchlist(ticker: str, db: Session = Depends(get_db)):
    a = universe_service.set_watchlist(db, ticker, False)
    return {"ticker": a.ticker, "watchlist": False}
