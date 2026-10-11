from dataclasses import replace
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.domain import importer
from app.domain.money import ZERO
from app.domain.portfolio import PositionError, TxnInput, build_portfolio
from app.domain.simulator import SCENARIOS, Assumptions, SimulationInput, extras_impact, required_contribution, scenarios, simulate, time_to_income
from app.models import ImportBatch, StrategyConfig, Transaction
from app.routers.portfolio import PID, ref_date
from app.schemas import ExtrasImpactIn, GoalContributionIn, GoalTimeIn, SimulationIn, StrategyIn, StrategyOut
from app.services import portfolio_service as svc

router = APIRouter(prefix="/api", tags=["planejamento, simulador e importação"])


# ------------------------------------------------------------------ estratégia
@router.get("/planning/strategy", response_model=StrategyOut)
def get_strategy(pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.get_strategy(db, pid, today)


@router.put("/planning/strategy", response_model=StrategyOut)
def put_strategy(body: StrategyIn, pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    s = svc.get_strategy(db, pid, today)
    for k in StrategyIn.model_fields:
        setattr(s, k, getattr(body, k))
    s.start_month = body.start_month.replace(day=1)
    s.extra_contributions = [e.model_dump(mode="json") for e in body.extra_contributions]  # JSON: valores como string
    db.commit()
    return s


@router.get("/planning/ledger")
def get_ledger(pid: int = PID, months_ahead: int = Query(1, ge=0, le=12), today: date = Depends(ref_date), db: Session = Depends(get_db)):
    return svc.ledger(db, pid, today, months_ahead)


# ------------------------------------------------------------------ simulador
def _sim_input(db: Session, pid: int, body: SimulationIn, today: date) -> tuple[SimulationInput, dict]:
    initial = body.initial_value
    meta = {}
    if initial is None:
        s = svc.summary(db, pid, today)
        initial = s["market_value"] if s["market_value_complete"] and s["market_value"] else s["invested_total"]
        meta["initial_value_source"] = "valor de mercado atual" if s["market_value_complete"] else "custo investido (cotações incompletas)"
        if "DEMO" in s["price_sources"]:
            meta["warning"] = "Atenção: o valor de mercado usa cotações DEMO (ilustrativas)."
    if body.assumptions:
        a = Assumptions(**body.assumptions.model_dump())
    else:
        if body.scenario not in SCENARIOS:
            raise HTTPException(422, f"Cenário desconhecido: {body.scenario}")
        a = SCENARIOS[body.scenario]
    inp = SimulationInput(
        initial_value=initial or ZERO,
        monthly_contribution=body.monthly_contribution,
        annual_contribution_increase=body.annual_contribution_increase,
        extras=[(e.month, e.amount) for e in body.extras],
        reinvest_pct=body.reinvest_pct,
        years=body.years,
        start=date(today.year, today.month, 1),
        assumptions=a,
    )
    return inp, meta


@router.get("/simulator/scenarios")
def list_scenarios():
    return {k: {f: str(getattr(v, f)) for f in v.__dataclass_fields__} for k, v in SCENARIOS.items()}


@router.post("/simulator/run")
def run_simulation(body: SimulationIn, pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    inp, meta = _sim_input(db, pid, body, today)
    overrides = {k: Assumptions(**v.model_dump()) for k, v in (body.scenario_overrides or {}).items()}
    return {"meta": meta, "scenarios": scenarios(inp, overrides)}


@router.post("/simulator/goal-time")
def goal_time(body: GoalTimeIn, pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    inp, meta = _sim_input(db, pid, body, today)
    return {"meta": meta, **time_to_income(inp, body.target_monthly_income, body.real)}


@router.post("/simulator/goal-contribution")
def goal_contribution(body: GoalContributionIn, pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    inp, meta = _sim_input(db, pid, body, today)
    return {"meta": meta, **required_contribution(inp, body.target_monthly_income, body.years, body.real)}


@router.post("/simulator/extras-impact")
def extras(body: ExtrasImpactIn, pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    inp, meta = _sim_input(db, pid, body, today)
    return {"meta": meta, **extras_impact(inp, [(e.month, e.amount) for e in body.compare_extras])}


# ------------------------------------------------------------------ importação
@router.get("/imports/template", response_class=PlainTextResponse)
def import_template():
    return PlainTextResponse(importer.template_csv(), headers={"Content-Disposition": "attachment; filename=investai_modelo.csv"})


def _classify(db: Session, pid: int, rows: list[importer.ParsedRow]):
    hashes = {r.hash for r in rows if r.hash}
    existing = set(
        db.scalars(select(Transaction.external_hash).where(Transaction.portfolio_id == pid, Transaction.external_hash.in_(hashes)))
    ) if hashes else set()
    # conciliação com lançamentos manuais (sem hash): mesma data, tipo, ativo e quantidade
    manual = {}
    for t in svc.load_transactions(db, pid):
        if t.external_hash is None:
            key = (t.trade_date, str(t.type), t.asset.ticker if t.asset else None, importer._num(t.quantity))
            manual[key] = manual.get(key, 0) + 1
    out = []
    for r in rows:
        if r.skipped_reason:
            status = "IGNORADA"
        elif r.errors:
            status = "ERRO"
        elif r.hash in existing:
            status = "DUPLICADA"
        else:
            key = (r.data["trade_date"], str(r.data["type"]), r.data.get("ticker"), importer._num(r.data.get("quantity")))
            if manual.get(key):
                manual[key] -= 1
                status = "POSSIVEL_DUPLICATA"
            else:
                status = "NOVA"
        out.append((r, status))
    return out


def _row_json(r: importer.ParsedRow, status: str) -> dict:
    d = {k: (str(v) if v is not None else None) for k, v in r.data.items()}
    return {"row": r.row_number, "status": status, "data": d, "errors": r.errors, "skipped_reason": r.skipped_reason}


@router.post("/imports/preview")
async def import_preview(file: UploadFile = File(...), fmt: str | None = Form(None), pid: int = PID, db: Session = Depends(get_db)):
    svc.ensure_portfolio(db, pid)
    content = await file.read()
    try:
        fmt, rows = importer.parse_file(content, file.filename or "upload.csv", fmt)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    classified = _classify(db, pid, rows)
    counts: dict[str, int] = {}
    for _, s in classified:
        counts[s] = counts.get(s, 0) + 1

    # Conciliação: posição atual vs. posição após importar as linhas NOVAS
    current = [svc.to_input(t) for t in svc.load_transactions(db, pid)]
    new = [TxnInput(type=r.data["type"], status=r.data["status"], trade_date=r.data["trade_date"], ticker=r.data.get("ticker"),
                    quantity=r.data.get("quantity"), unit_price=r.data.get("unit_price"), fees=r.data.get("fees") or ZERO,
                    total_amount=r.data.get("total_amount"), ratio_from=r.data.get("ratio_from"), ratio_to=r.data.get("ratio_to"))
           for r, s in classified if s == "NOVA"]
    reconciliation, error = [], None
    try:
        before = build_portfolio(current).positions
        after = build_portfolio(current + new).positions
        for t in sorted(set(before) | set(after)):
            b = before[t].quantity if t in before else ZERO
            a = after[t].quantity if t in after else ZERO
            if a != b:
                reconciliation.append({"ticker": t, "before": str(b), "after": str(a), "delta": str(a - b)})
    except PositionError as e:
        error = f"Importação geraria posição inconsistente: {e}"
    return {
        "format": fmt,
        "filename": file.filename,
        "counts": counts,
        "rows": [_row_json(r, s) for r, s in classified],
        "reconciliation": reconciliation,
        "consistency_error": error,
    }


@router.post("/imports/commit")
async def import_commit(
    file: UploadFile = File(...),
    fmt: str | None = Form(None),
    include_possible_duplicates: bool = Form(False),
    pid: int = PID,
    db: Session = Depends(get_db),
):
    svc.ensure_portfolio(db, pid)
    content = await file.read()
    try:
        fmt, rows = importer.parse_file(content, file.filename or "upload.csv", fmt)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    classified = _classify(db, pid, rows)
    accept = {"NOVA"} | ({"POSSIVEL_DUPLICATA"} if include_possible_duplicates else set())
    batch = ImportBatch(portfolio_id=pid, filename=file.filename or "upload", fmt=fmt, rows_total=len(rows))
    db.add(batch)
    db.flush()
    imported = 0
    for r, s in classified:
        if s not in accept:
            continue
        d = dict(r.data)
        ticker = d.pop("ticker", None)
        t = Transaction(portfolio_id=pid, source=f"IMPORT:{fmt}", external_hash=r.hash, import_batch_id=batch.id, **d)
        t.asset = svc.get_asset(db, ticker) if ticker else None
        db.add(t)
        imported += 1
    batch.rows_imported = imported
    batch.rows_duplicate = sum(1 for _, s in classified if s in ("DUPLICADA", "POSSIVEL_DUPLICATA") and s not in accept)
    batch.rows_error = sum(1 for _, s in classified if s == "ERRO")
    db.flush()
    try:
        svc.check_consistency([svc.to_input(x) for x in svc.load_transactions(db, pid)])
    except HTTPException:
        db.rollback()
        raise
    db.commit()
    return {"batch_id": batch.id, "format": fmt, "imported": imported, "duplicates_skipped": batch.rows_duplicate, "errors": batch.rows_error}
