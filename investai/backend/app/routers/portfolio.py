from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.domain.enums import TxnStatus
from app.models import Asset, Transaction
from app.schemas import AssetIn, AssetOut, AssetPatch, ConfirmIn, TransactionIn, TransactionOut
from app.services import portfolio_service as svc

router = APIRouter(prefix="/api", tags=["carteira"])
PID = Query(svc.DEFAULT_PORTFOLIO_ID, alias="portfolio_id")


def ref_date(as_of: date | None = Query(None, description="Data de referência (padrão: hoje, America/Sao_Paulo)")) -> date:
    return as_of or svc.today_br()


# ------------------------------------------------------------------ ativos
@router.get("/assets", response_model=list[AssetOut])
def list_assets(db: Session = Depends(get_db)):
    return db.scalars(select(Asset).order_by(Asset.ticker)).all()


@router.post("/assets", response_model=AssetOut, status_code=201)
def create_asset(body: AssetIn, db: Session = Depends(get_db)):
    if db.scalar(select(Asset).where(Asset.ticker == body.ticker)):
        raise HTTPException(409, "Ativo já cadastrado")
    a = Asset(**body.model_dump())
    db.add(a)
    db.commit()
    return a


@router.patch("/assets/{ticker}", response_model=AssetOut)
def update_asset(ticker: str, body: AssetPatch, db: Session = Depends(get_db)):
    a = svc.get_asset(db, ticker, create=False)
    if not a:
        raise HTTPException(404, "Ativo não encontrado")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(a, k, v)
    db.commit()
    return a


# ------------------------------------------------------------------ operações
@router.get("/transactions", response_model=list[TransactionOut])
def list_transactions(
    pid: int = PID,
    ticker: str | None = None,
    status: TxnStatus | None = None,
    db: Session = Depends(get_db),
):
    svc.ensure_portfolio(db, pid)
    stmt = select(Transaction).where(Transaction.portfolio_id == pid)
    if ticker:
        a = svc.get_asset(db, ticker, create=False)
        if not a:
            return []
        stmt = stmt.where(Transaction.asset_id == a.id)
    if status:
        stmt = stmt.where(Transaction.status == status)
    return db.scalars(stmt.order_by(Transaction.trade_date.desc(), Transaction.id.desc())).unique().all()


def _apply_body(db: Session, t: Transaction, body: TransactionIn):
    data = body.model_dump()
    ticker = data.pop("ticker")
    t.asset = svc.get_asset(db, ticker) if ticker else None
    for k, v in data.items():
        setattr(t, k, v)


def _validate_portfolio(db: Session, pid: int):
    db.flush()
    db.expire_all()
    svc.check_consistency([svc.to_input(x) for x in svc.load_transactions(db, pid)])


@router.post("/transactions", response_model=TransactionOut, status_code=201)
def create_transaction(body: TransactionIn, pid: int = PID, db: Session = Depends(get_db)):
    svc.ensure_portfolio(db, pid)
    t = Transaction(portfolio_id=pid, source="MANUAL")
    _apply_body(db, t, body)
    db.add(t)
    try:
        _validate_portfolio(db, pid)
    except HTTPException:
        db.rollback()
        raise
    db.commit()
    return db.get(Transaction, t.id)


@router.put("/transactions/{txn_id}", response_model=TransactionOut)
def update_transaction(txn_id: int, body: TransactionIn, db: Session = Depends(get_db)):
    t = db.get(Transaction, txn_id)
    if not t:
        raise HTTPException(404, "Operação não encontrada")
    _apply_body(db, t, body)
    try:
        _validate_portfolio(db, t.portfolio_id)
    except HTTPException:
        db.rollback()
        raise
    db.commit()
    return db.get(Transaction, txn_id)


@router.post("/transactions/{txn_id}/confirm", response_model=TransactionOut)
def confirm_transaction(txn_id: int, body: ConfirmIn, db: Session = Depends(get_db)):
    """Converte PENDING/PLANNED em CONFIRMED (ex.: recibo de subscrição convertido, ordem executada)."""
    t = db.get(Transaction, txn_id)
    if not t:
        raise HTTPException(404, "Operação não encontrada")
    if t.status == TxnStatus.CONFIRMED:
        raise HTTPException(409, "Operação já confirmada")
    for k, v in body.model_dump(exclude_unset=True, exclude_none=True).items():
        setattr(t, k, v)
    if t.unit_price is None and t.total_amount is None:
        raise HTTPException(422, "Informe o preço executado (unit_price) ou o valor total para confirmar.")
    t.status = TxnStatus.CONFIRMED
    try:
        _validate_portfolio(db, t.portfolio_id)
    except HTTPException:
        db.rollback()
        raise
    db.commit()
    return db.get(Transaction, txn_id)


@router.delete("/transactions/{txn_id}", status_code=204)
def delete_transaction(txn_id: int, db: Session = Depends(get_db)):
    t = db.get(Transaction, txn_id)
    if not t:
        raise HTTPException(404, "Operação não encontrada")
    pid = t.portfolio_id
    db.delete(t)
    try:
        _validate_portfolio(db, pid)
    except HTTPException:
        db.rollback()
        raise
    db.commit()


# ------------------------------------------------------------------ visões da carteira
@router.get("/portfolio/positions")
def positions(pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.positions_view(db, pid, today)


@router.get("/portfolio/summary")
def portfolio_summary(pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.summary(db, pid, today)


@router.get("/portfolio/allocation")
def portfolio_allocation(pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.allocation(db, pid, today)


@router.get("/portfolio/history")
def portfolio_history(pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.history(db, pid, today)


@router.get("/dashboard")
def get_dashboard(pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.dashboard(db, pid, today)


@router.get("/alerts")
def get_alerts(pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.alerts(db, pid, today)
