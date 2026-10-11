"""Serviços de carteira: ponte entre ORM e o domínio puro, e composição das visões da API."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.dividends import (
    AnnouncementInput,
    add_months,
    dividend_stats,
    income_overview,
    month_key,
    received_txns_from,
)
from app.domain.enums import AssetClass, TxnStatus, TxnType
from app.domain.money import ZERO, D, TWELVE, money, safe_div
from app.domain.planning import ExtraContribution, StrategyInput, affordable, budget_ledger
from app.domain.portfolio import PortfolioState, PositionError, TxnInput, build_portfolio, unrealized
from app.models import Asset, DividendAnnouncement, Portfolio, PriceQuote, StrategyConfig, Transaction, User

DEFAULT_PORTFOLIO_ID = 1
BR_TZ = ZoneInfo("America/Sao_Paulo")


def brl(v) -> str:
    """R$ 1.234,56"""
    return "R$ " + f"{money(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def qty_br(v) -> str:
    return format(D(v).normalize(), "f").replace(".", ",")


def today_br() -> date:
    return datetime.now(BR_TZ).date()


# ---------------------------------------------------------------- carregamento
def to_input(t: Transaction) -> TxnInput:
    return TxnInput(
        id=t.id or 0,
        type=TxnType(t.type),
        status=TxnStatus(t.status),
        trade_date=t.trade_date,
        ticker=t.asset.ticker if t.asset else None,
        quantity=t.quantity,
        unit_price=t.unit_price,
        fees=t.fees or ZERO,
        total_amount=t.total_amount,
        ratio_from=t.ratio_from,
        ratio_to=t.ratio_to,
    )


def load_transactions(db: Session, pid: int) -> list[Transaction]:
    return list(db.scalars(select(Transaction).where(Transaction.portfolio_id == pid).order_by(Transaction.trade_date, Transaction.id)))


def ensure_portfolio(db: Session, pid: int) -> Portfolio:
    p = db.get(Portfolio, pid)
    if p is None:
        if pid != DEFAULT_PORTFOLIO_ID:
            raise HTTPException(404, "Carteira não encontrada")
        user = db.scalar(select(User).limit(1)) or User(email="local@investai.dev", name="Investidor")
        db.add(user)
        db.flush()
        p = Portfolio(id=DEFAULT_PORTFOLIO_ID, user_id=user.id, name="Carteira principal")
        db.add(p)
        db.commit()
    return p


def get_asset(db: Session, ticker: str, create: bool = True) -> Asset | None:
    ticker = ticker.strip().upper()
    a = db.scalar(select(Asset).where(Asset.ticker == ticker))
    if a is None and create:
        cls = AssetClass.FII if ticker.endswith("11") else AssetClass.STOCK
        a = Asset(ticker=ticker, asset_class=cls, segment=None)
        db.add(a)
        db.flush()
    return a


def check_consistency(inputs: list[TxnInput]) -> PortfolioState:
    """Replays the full history; rejects changes that would produce inconsistent positions."""
    try:
        return build_portfolio(inputs)
    except PositionError as e:
        raise HTTPException(422, str(e)) from e


@dataclass
class QuoteInfo:
    price: object
    quote_date: date
    source: str


def latest_quotes(db: Session, on: date | None = None) -> dict[str, QuoteInfo]:
    stmt = select(PriceQuote, Asset.ticker).join(Asset, Asset.id == PriceQuote.asset_id)
    if on:
        stmt = stmt.where(PriceQuote.quote_date <= on)
    stmt = stmt.order_by(PriceQuote.quote_date, PriceQuote.fetched_at)
    out: dict[str, QuoteInfo] = {}
    for q, ticker in db.execute(stmt):
        out[ticker] = QuoteInfo(q.close, q.quote_date, q.source)
    return out


def load_announcements(db: Session) -> list[AnnouncementInput]:
    rows = db.scalars(select(DividendAnnouncement))
    return [
        AnnouncementInput(
            id=a.id, ticker=a.asset.ticker, amount_per_share=a.amount_per_share, com_date=a.com_date,
            payment_date=a.payment_date, ex_date=a.ex_date, kind=a.kind, source=a.source,
        )
        for a in rows
    ]


def _context(db: Session, pid: int):
    ensure_portfolio(db, pid)
    txns = load_transactions(db, pid)
    inputs = [to_input(t) for t in txns]
    links = {t.id: t.announcement_id for t in txns}
    return txns, inputs, links


# ---------------------------------------------------------------- posições
def positions_view(db: Session, pid: int, today: date) -> dict:
    txns, inputs, links = _context(db, pid)
    state = check_consistency(inputs)
    quotes = latest_quotes(db)
    anns = load_announcements(db)
    received = received_txns_from(inputs, links)
    assets = {a.ticker: a for a in db.scalars(select(Asset))}

    rows = []
    for ticker, p in sorted(state.positions.items()):
        if not p.is_open and p.realized_pnl == 0 and p.dividends_received == 0:
            continue
        q = quotes.get(ticker)
        price = q.price if q else None
        u = unrealized(p, price)
        stats = dividend_stats(ticker, anns, received, inputs, today, price=price, avg_price=p.avg_price)
        a = assets.get(ticker)
        rows.append(
            {
                "ticker": ticker,
                "name": a.name if a else None,
                "segment": a.segment if a else None,
                "asset_class": a.asset_class if a else None,
                "quantity": p.quantity,
                "avg_price": p.avg_price,
                "total_cost": money(p.total_cost) if p.cost_known else None,
                "cost_known": p.cost_known,
                "pending_quantity": p.pending_quantity,
                "pending_cost": money(p.pending_cost),
                "planned_quantity": p.planned_quantity,
                "planned_cost": money(p.planned_cost),
                "price": price,
                "price_date": q.quote_date if q else None,
                "price_source": q.source if q else None,
                **u,
                "realized_pnl": money(p.realized_pnl),
                "dividends_received": money(p.dividends_received),
                "amortizations_received": money(p.amortizations_received),
                "dividend_stats": {k: stats[k] for k in ("last", "sum_12m", "mean_monthly_12m", "dy_12m", "dy_last_annualized", "yoc_12m", "regularity_12m")},
            }
        )
    _add_weights(rows)
    return {"reference_date": today, "positions": rows, "warnings": state.warnings, "cash": _cash(state)}


def _cash(state: PortfolioState) -> dict:
    c = state.cash
    return {k: money(getattr(c, k)) for k in ("deposits", "withdrawals", "purchases", "sales", "dividends", "amortizations", "fees")} | {
        "balance": money(c.balance)
    }


def _add_weights(rows: list[dict]) -> str:
    held = [r for r in rows if r["quantity"] > 0]
    if held and all(r["market_value"] is not None for r in held):
        basis, key = "MERCADO", "market_value"
    elif held and all(r["cost_known"] for r in held):
        basis, key = "CUSTO", "total_cost"
    else:
        basis, key = "MISTO", None
    vals = {}
    for r in held:
        if key:
            vals[r["ticker"]] = D(r[key])
        else:
            vals[r["ticker"]] = D(r["market_value"] if r["market_value"] is not None else (r["total_cost"] or 0))
    total = sum(vals.values(), ZERO)
    for r in rows:
        r["weight"] = safe_div(vals.get(r["ticker"], ZERO), total) if total else None
        r["weight_basis"] = basis
    return basis


def allocation(db: Session, pid: int, today: date) -> dict:
    pv = positions_view(db, pid, today)
    rows = [r for r in pv["positions"] if r["quantity"] > 0]
    basis = rows[0]["weight_basis"] if rows else "MERCADO"
    by_seg, by_cls = defaultdict(lambda: ZERO), defaultdict(lambda: ZERO)
    for r in rows:
        w = r["weight"] or ZERO
        by_seg[r["segment"] or "Não classificado"] += w
        by_cls[str(r["asset_class"] or "N/D")] += w
    notes = []
    if basis != "MERCADO":
        notes.append("Distribuição calculada por custo (ou mista) porque há ativos sem cotação registrada.")
    return {
        "basis": basis,
        "by_asset": [{"label": r["ticker"], "weight": r["weight"]} for r in sorted(rows, key=lambda r: -(r["weight"] or 0))],
        "by_segment": [{"label": k, "weight": v} for k, v in sorted(by_seg.items(), key=lambda kv: -kv[1])],
        "by_class": [{"label": k, "weight": v} for k, v in sorted(by_cls.items(), key=lambda kv: -kv[1])],
        "notes": notes,
    }


# ---------------------------------------------------------------- resumo
def summary(db: Session, pid: int, today: date) -> dict:
    pv = positions_view(db, pid, today)
    rows = pv["positions"]
    held = [r for r in rows if r["quantity"] > 0]
    _, inputs, _ = _context(db, pid)

    cost_unknown = [r["ticker"] for r in held if not r["cost_known"]]
    missing_prices = [r["ticker"] for r in held if r["price"] is None]
    confirmed_cost = sum((D(r["total_cost"]) for r in held if r["cost_known"]), ZERO)
    pending_cost = sum((D(r["pending_cost"]) for r in rows), ZERO)
    market_value = sum((D(r["market_value"]) for r in held if r["market_value"] is not None), ZERO)
    comparable = [r for r in held if r["unrealized_pnl"] is not None]
    unreal = sum((D(r["unrealized_pnl"]) for r in comparable), ZERO)
    comparable_cost = sum((D(r["total_cost"]) for r in comparable), ZERO)

    div = [t for t in inputs if t.type == TxnType.DIVIDEND and t.status == TxnStatus.CONFIRMED]
    month_start = date(today.year, today.month, 1)
    last_month_start = add_months(month_start, -1)
    d_month = sum((D(t.total_amount) for t in div if month_start <= t.trade_date <= today), ZERO)
    d_last_month = sum((D(t.total_amount) for t in div if last_month_start <= t.trade_date < month_start), ZERO)
    d_ytd = sum((D(t.total_amount) for t in div if t.trade_date.year == today.year and t.trade_date <= today), ZERO)
    d_12m = sum((D(t.total_amount) for t in div if add_months(month_start, -11) <= t.trade_date <= today), ZERO)
    d_total = sum((D(t.total_amount) for t in div), ZERO)
    realized = sum((D(r["realized_pnl"]) for r in rows), ZERO)

    state = build_portfolio(inputs)
    complete = not missing_prices and not cost_unknown
    total_return = None
    notes = []
    if complete:
        opening_cost = ZERO
        for t in inputs:
            if t.type == TxnType.OPENING_BALANCE and t.status == TxnStatus.CONFIRMED and t.gross_cost() is not None:
                opening_cost += t.gross_cost()
        invested_ever = state.cash.purchases + opening_cost
        received = state.cash.sales + state.cash.dividends + state.cash.amortizations
        gain = market_value + pending_cost + received - invested_ever - state.cash.fees
        total_return = {
            "absolute": money(gain),
            "pct": safe_div(gain, invested_ever),
            "method": "Retorno simples: (valor de mercado + recibos pendentes a custo + vendas + proventos + amortizações − capital aplicado − taxas) / capital aplicado.",
        }
    else:
        notes.append("Rentabilidade total indisponível enquanto houver ativos sem preço médio ou sem cotação.")
    if cost_unknown:
        notes.append(f"Preço médio não informado: {', '.join(cost_unknown)}. Edite o saldo inicial em Movimentações.")
    if missing_prices:
        notes.append(f"Sem cotação registrada: {', '.join(missing_prices)}. Valor de mercado parcial.")

    return {
        "reference_date": today,
        "invested_confirmed": money(confirmed_cost),
        "invested_pending": money(pending_cost),
        "invested_total": money(confirmed_cost + pending_cost),
        "market_value": money(market_value) if held else ZERO,
        "market_value_complete": not missing_prices,
        "unrealized_pnl": money(unreal),
        "unrealized_pct": safe_div(unreal, comparable_cost),
        "realized_pnl": money(realized),
        "dividends_month": money(d_month),
        "dividends_last_month": money(d_last_month),
        "dividends_ytd": money(d_ytd),
        "dividends_12m": money(d_12m),
        "dividends_total": money(d_total),
        "dy_last_month": safe_div(d_last_month, market_value) if market_value else None,
        "dy_last_month_annualized": (safe_div(d_last_month, market_value) * TWELVE) if market_value and d_last_month else None,
        "dy_12m": safe_div(d_12m, market_value) if market_value else None,
        "yoc_12m": safe_div(d_12m, confirmed_cost) if confirmed_cost and not cost_unknown else None,
        "total_return": total_return,
        "positions_count": len(held),
        "cost_unknown": cost_unknown,
        "missing_prices": missing_prices,
        "price_sources": sorted({r["price_source"] for r in held if r["price_source"]}),
        "benchmarks": {
            "available": False,
            "reason": "Comparação com CDI, IFIX e IPCA requer as séries do BCB/IBGE/B3 (Fase 2). Adaptador BCB já implementado.",
        },
        "notes": notes,
        "definitions": {
            "dy_last_month_annualized": "Proventos recebidos no mês anterior ÷ valor de mercado atual × 12 (anualização simples).",
            "dy_12m": "Proventos recebidos nos últimos 12 meses ÷ valor de mercado atual.",
            "yoc_12m": "Proventos recebidos nos últimos 12 meses ÷ custo de aquisição confirmado.",
        },
    }


def history(db: Session, pid: int, today: date) -> list[dict]:
    _, inputs, _ = _context(db, pid)
    executed = [t for t in inputs if t.status != TxnStatus.PLANNED]
    if not executed:
        return []
    first = min(t.trade_date for t in executed)
    m = date(first.year, first.month, 1)
    end = date(today.year, today.month, 1)
    out, cum_div = [], ZERO
    while m <= end:
        month_end = add_months(m, 1).replace(day=1)
        last_day = date.fromordinal(month_end.toordinal() - 1)
        as_of = min(last_day, today)
        st = build_portfolio(executed, as_of=as_of)
        invested = sum((p.total_cost + p.pending_cost for p in st.positions.values()), ZERO)
        div_m = sum(
            (D(t.total_amount) for t in executed if t.type == TxnType.DIVIDEND and m <= t.trade_date <= as_of), ZERO
        )
        cum_div += div_m
        quotes = latest_quotes(db, on=as_of)
        held = [p for p in st.positions.values() if p.quantity > 0]
        mv = None
        if held and all(p.ticker in quotes for p in held):
            mv = money(sum((p.quantity * D(quotes[p.ticker].price) for p in held), ZERO))
        out.append(
            {"month": month_key(m), "invested": money(invested), "market_value": mv, "dividends": money(div_m), "dividends_cum": money(cum_div)}
        )
        m = add_months(m, 1)
    return out


# ---------------------------------------------------------------- proventos
def income(db: Session, pid: int, today: date, horizon_months: int = 12) -> dict:
    _, inputs, links = _context(db, pid)
    return income_overview(inputs, received_txns_from(inputs, links), load_announcements(db), today, horizon_months)


def stats_for(db: Session, pid: int, ticker: str, today: date) -> dict:
    _, inputs, links = _context(db, pid)
    state = build_portfolio(inputs)
    p = state.positions.get(ticker)
    q = latest_quotes(db).get(ticker)
    return dividend_stats(
        ticker, load_announcements(db), received_txns_from(inputs, links), inputs, today,
        price=q.price if q else None, avg_price=p.avg_price if p else None,
    )


# ---------------------------------------------------------------- planejamento
def get_strategy(db: Session, pid: int, today: date) -> StrategyConfig:
    ensure_portfolio(db, pid)
    s = db.scalar(select(StrategyConfig).where(StrategyConfig.portfolio_id == pid))
    if s is None:
        s = StrategyConfig(portfolio_id=pid, start_month=date(today.year, today.month, 1), extra_contributions=[])
        db.add(s)
        db.commit()
    return s


def strategy_input(s: StrategyConfig) -> StrategyInput:
    return StrategyInput(
        monthly_contribution=s.monthly_contribution,
        reinvest_pct=s.reinvest_pct,
        annual_increase_pct=s.annual_increase_pct,
        extras=[ExtraContribution(month=int(e["month"]), amount=D(e["amount"]), label=e.get("label", "")) for e in s.extra_contributions or []],
        start_month=s.start_month,
        initial_cash=s.initial_cash,
    )


def ledger(db: Session, pid: int, today: date, months_ahead: int = 1) -> dict:
    s = get_strategy(db, pid, today)
    _, inputs, _ = _context(db, pid)
    quotes = latest_quotes(db)
    rows = budget_ledger(strategy_input(s), inputs, today, months_ahead, prices={t: q.price for t, q in quotes.items()})
    current = next((r for r in rows if r["is_current"]), None) or (rows[0] if rows else None)
    assets = db.scalars(select(Asset).where(Asset.is_active.is_(True)))
    prices = {a.ticker: quotes[a.ticker].price for a in assets if a.ticker in quotes}
    return {
        "rows": rows,
        "current": current,
        "affordable": affordable(current["available"], prices) if current else [],
        "notes": [
            "Aporte próprio: usa os aportes registrados no mês (APORTE/RETIRADA); sem registro, usa o valor planejado.",
            "Compras confirmadas e pendentes (já pagas) consomem o orçamento; ordens planejadas apenas aparecem como comprometidas (sem preço informado, estimadas pela última cotação).",
            "A lista 'cabe no orçamento' é aritmética simples, não uma recomendação de investimento.",
        ],
    }


# ---------------------------------------------------------------- alertas
def alerts(db: Session, pid: int, today: date) -> list[dict]:
    from app.core.config import get_settings

    cfg = get_settings()
    s = get_strategy(db, pid, today)
    pv = positions_view(db, pid, today)
    out: list[dict] = []

    def add(severity, kind, title, message):
        out.append({"severity": severity, "type": kind, "title": title, "message": message})

    for r in pv["positions"]:
        if r["quantity"] > 0 and r["weight"] is not None and r["weight"] > D(s.concentration_limit_asset):
            add("warning", "CONCENTRACAO_ATIVO", f"Concentração em {r['ticker']}",
                f"{r['ticker']} representa {str(round(r['weight'] * 100, 1)).replace('.', ',')}% da carteira (limite {D(s.concentration_limit_asset) * 100:.0f}%, base {r['weight_basis'].lower()}).")
        if r["pending_quantity"] > 0:
            add("info", "SUBSCRICAO_PENDENTE", f"Subscrição pendente em {r['ticker']}",
                f"{qty_br(r['pending_quantity'])} cotas aguardando liquidação/conversão. Confirme em Movimentações quando creditadas.")
        if r["planned_quantity"] > 0:
            add("info", "ORDEM_PLANEJADA", f"Ordem planejada de {r['ticker']}",
                f"Compra planejada de {qty_br(r['planned_quantity'])} cotas ainda não executada.")
        if r["quantity"] > 0 and r["price"] is not None and r["price_date"] and r["quantity"] > 0 and (today - r["price_date"]).days > cfg.stale_quote_days:
            add("info", "COTACAO_DESATUALIZADA", f"Cotação de {r['ticker']} desatualizada",
                f"Última cotação em {r['price_date'].strftime('%d/%m/%Y')} ({r['price_source']}).")

    held = [r for r in pv["positions"] if r["quantity"] > 0]
    no_cost = [r["ticker"] for r in held if not r["cost_known"]]
    if no_cost:
        add("warning", "PRECO_MEDIO_AUSENTE", f"Preço médio não informado ({len(no_cost)})",
            f"{', '.join(no_cost)}: sem preço médio não há lucro/prejuízo nem Yield on Cost. Edite o saldo inicial em Movimentações.")
    no_price = [r["ticker"] for r in held if r["price"] is None]
    if no_price:
        add("warning", "SEM_COTACAO", f"Sem cotação ({len(no_price)})",
            f"{', '.join(no_price)}: registre a cotação manualmente ou configure um provedor de mercado.")

    seg = allocation(db, pid, today)
    for item in seg["by_segment"]:
        if item["weight"] > D(s.concentration_limit_segment):
            add("warning", "CONCENTRACAO_SEGMENTO", f"Concentração no segmento {item['label']}",
                f"{str(round(item['weight'] * 100, 1)).replace('.', ',')}% da carteira (limite {D(s.concentration_limit_segment) * 100:.0f}%).")

    inc = income(db, pid, today, horizon_months=1)
    soon = [c for c in inc["confirmed"] if c["status"] == "A_RECEBER" and 0 <= (c["payment_date"] - today).days <= cfg.upcoming_payment_days]
    if soon:
        total = sum((c["amount"] for c in soon), ZERO)
        first = min(c["payment_date"] for c in soon)
        add("success", "PAGAMENTO_PROXIMO", f"{len(soon)} pagamento(s) nos próximos {cfg.upcoming_payment_days} dias",
            f"{brl(total)} confirmados ({', '.join(sorted({c['ticker'] for c in soon}))}); o primeiro em {first.strftime('%d/%m/%Y')}.")
    unreg = [c for c in inc["confirmed"] if c["status"] == "PAGO_NAO_REGISTRADO"]
    if unreg:
        add("info", "PAGAMENTO_NAO_REGISTRADO", f"{len(unreg)} provento(s) pago(s) e não registrado(s)",
            f"{', '.join(sorted({c['ticker'] for c in unreg}))}: registre como recebido na tela Dividendos.")

    lg = ledger(db, pid, today, months_ahead=0)
    if lg["current"] and lg["affordable"]:
        can = [a for a in lg["affordable"] if a["max_quantity"] > 0]
        if can:
            add("success", "ORCAMENTO_DISPONIVEL", "Orçamento disponível para aporte",
                f"{brl(lg['current']['available'])} disponíveis em {lg['current']['month']}.")
    if today.month in (1, 4, 7, 10) and today.day <= 15:
        add("info", "REVISAO_TRIMESTRAL", "Revisão trimestral da carteira", "Início de trimestre: revise concentração, proventos e estratégia.")
    order = {"warning": 0, "success": 1, "info": 2}
    return sorted(out, key=lambda a: order.get(a["severity"], 9))


# ---------------------------------------------------------------- dashboard
def dashboard(db: Session, pid: int, today: date) -> dict:
    inc = income(db, pid, today, horizon_months=3)
    mk = month_key(today)
    month_income = {
        "received": money(sum((r["amount"] for r in inc["received"] if month_key(r["payment_date"]) == mk), ZERO)),
        "confirmed": money(sum((c["amount"] for c in inc["confirmed"] if month_key(c["payment_date"]) == mk and c["status"] == "A_RECEBER"), ZERO)),
        "estimated": money(sum((e["amount"] for e in inc["estimated"] if e["month"] == mk), ZERO)),
    }
    upcoming = [c for c in inc["confirmed"] if c["status"] == "A_RECEBER"][:10]
    hist = history(db, pid, today)
    return {
        "reference_date": today,
        "summary": summary(db, pid, today),
        "month_income": month_income,
        "upcoming_payments": upcoming,
        "budget": ledger(db, pid, today, months_ahead=0)["current"],
        "allocation": allocation(db, pid, today),
        "dividends_history": hist[-12:],
        "alerts": alerts(db, pid, today)[:8],
    }
