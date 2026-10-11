"""Proventos: elegibilidade por data-com, categorias de renda e estatísticas de distribuição.

Categorias de renda (nunca misturadas):
- RECEBIDA: pagamento efetivamente registrado (operação DIVIDEND).
- CONFIRMADA: anúncio oficial com valor por cota; quantidade elegível = cotas confirmadas ao final da data-com.
- ESTIMADA: média histórica por cota × posição atual. É premissa, não promessa.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from app.domain.enums import DividendKind, TxnStatus, TxnType
from app.domain.money import ZERO, D, TWELVE, money, safe_div
from app.domain.portfolio import TxnInput, build_portfolio, quantity_on


@dataclass(frozen=True)
class AnnouncementInput:
    id: int
    ticker: str
    amount_per_share: object
    com_date: date
    payment_date: date
    ex_date: date | None = None
    kind: DividendKind = DividendKind.RENDIMENTO
    source: str = "MANUAL"


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


def months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


@dataclass(frozen=True)
class DividendTxn:
    id: int
    ticker: str
    payment_date: date
    amount: object
    quantity: object = None
    announcement_id: int | None = None


def _match_received(a: AnnouncementInput, received: list[DividendTxn]) -> DividendTxn | None:
    for r in received:
        if r.announcement_id == a.id:
            return r
    for r in received:
        if r.announcement_id is None and r.ticker == a.ticker and r.payment_date == a.payment_date:
            return r
    return None


def income_overview(
    txns: list[TxnInput],
    received: list[DividendTxn],
    announcements: list[AnnouncementInput],
    today: date,
    horizon_months: int = 12,
    estimation_window: int = 12,
) -> dict:
    current = build_portfolio([t for t in txns if t.status == TxnStatus.CONFIRMED], include_pending=False)

    received_items = [
        {
            "category": "RECEBIDA",
            "ticker": r.ticker,
            "payment_date": r.payment_date,
            "amount": money(r.amount),
            "quantity": r.quantity,
            "transaction_id": r.id,
        }
        for r in received
    ]

    confirmed_items = []
    covered: set[tuple[str, str]] = {(r.ticker, month_key(r.payment_date)) for r in received}
    for a in announcements:
        if _match_received(a, received):
            continue
        provisional = a.com_date > today
        if provisional:
            p = current.positions.get(a.ticker)
            qty = p.quantity if p else ZERO
        else:
            qty = quantity_on(txns, a.ticker, a.com_date)
        if qty <= 0:
            continue
        status = "A_RECEBER" if a.payment_date >= today else "PAGO_NAO_REGISTRADO"
        confirmed_items.append(
            {
                "category": "CONFIRMADA",
                "status": status,
                "kind": a.kind,
                "ticker": a.ticker,
                "com_date": a.com_date,
                "ex_date": a.ex_date,
                "payment_date": a.payment_date,
                "amount_per_share": D(a.amount_per_share),
                "eligible_quantity": qty,
                "amount": money(qty * D(a.amount_per_share)),
                "provisional_quantity": provisional,
                "announcement_id": a.id,
                "source": a.source,
            }
        )
        covered.add((a.ticker, month_key(a.payment_date)))

    estimated_items = []
    notes = [
        "Renda ESTIMADA = média mensal por cota dos proventos (RENDIMENTO) pagos nos últimos "
        f"{estimation_window} meses × cotas confirmadas atuais. Não é garantia de pagamento futuro.",
    ]
    first_month = date(today.year, today.month, 1)
    for ticker, pos in current.positions.items():
        if pos.quantity <= 0:
            continue
        mean = mean_monthly_per_share(ticker, announcements, received, txns, today, estimation_window)
        if mean is None:
            notes.append(f"{ticker}: sem histórico de proventos suficiente para estimar.")
            continue
        for i in range(horizon_months):
            m = add_months(first_month, i)
            if (ticker, month_key(m)) in covered:
                continue
            estimated_items.append(
                {
                    "category": "ESTIMADA",
                    "ticker": ticker,
                    "month": month_key(m),
                    "amount_per_share": mean,
                    "quantity": pos.quantity,
                    "amount": money(mean * pos.quantity),
                }
            )

    calendar: dict[str, dict] = defaultdict(lambda: {"RECEBIDA": ZERO, "CONFIRMADA": ZERO, "ESTIMADA": ZERO})
    for r in received_items:
        calendar[month_key(r["payment_date"])]["RECEBIDA"] += r["amount"]
    for c in confirmed_items:
        if c["kind"] == DividendKind.RENDIMENTO:
            calendar[month_key(c["payment_date"])]["CONFIRMADA"] += c["amount"]
    for e in estimated_items:
        calendar[e["month"]]["ESTIMADA"] += e["amount"]

    return {
        "reference_date": today,
        "received": sorted(received_items, key=lambda x: x["payment_date"], reverse=True),
        "confirmed": sorted(confirmed_items, key=lambda x: x["payment_date"]),
        "estimated": estimated_items,
        "calendar": [{"month": k, **v} for k, v in sorted(calendar.items())],
        "notes": notes,
    }


def _per_share_history(ticker, announcements, received, txns) -> list[tuple[date, object]]:
    """Valores por cota (data de pagamento). Prioriza anúncios; usa recebidos com quantidade como fallback."""
    hist = [
        (a.payment_date, D(a.amount_per_share))
        for a in announcements
        if a.ticker == ticker and a.kind == DividendKind.RENDIMENTO
    ]
    if hist:
        return sorted(hist)
    out = []
    for r in received:
        if r.ticker != ticker:
            continue
        qty = D(r.quantity) if r.quantity else None
        if not qty:
            continue
        out.append((r.payment_date, D(r.amount) / qty))
    return sorted(out)


def mean_monthly_per_share(ticker, announcements, received, txns, today: date, window: int = 12):
    hist = _per_share_history(ticker, announcements, received, txns)
    start = add_months(date(today.year, today.month, 1), -(window - 1))
    # inclui pagamentos já anunciados para o mês corrente (valor oficial conhecido)
    end = add_months(date(today.year, today.month, 1), 1)
    in_window = [(d, v) for d, v in hist if start <= d < end]
    if not in_window:
        return None
    first = min(d for d, _ in in_window)
    observed = min(window, months_between(first, today) + 1)
    return sum((v for _, v in in_window), ZERO) / D(max(observed, 1))


def _stdev(values: list) -> object:
    if len(values) < 2:
        return ZERO
    mean = sum(values, ZERO) / D(len(values))
    var = sum(((v - mean) ** 2 for v in values), ZERO) / D(len(values) - 1)
    return var.sqrt()


def dividend_stats(ticker: str, announcements, received, txns, today: date, price=None, avg_price=None) -> dict:
    hist = _per_share_history(ticker, announcements, received, txns)
    anns = sorted([a for a in announcements if a.ticker == ticker], key=lambda a: a.payment_date)
    last = anns[-1] if anns else None

    def window(months: int):
        start = add_months(date(today.year, today.month, 1), -(months - 1))
        end = add_months(date(today.year, today.month, 1), 1)
        return [(d, v) for d, v in hist if start <= d < end]

    h12, h24 = window(12), window(24)
    vals12 = [v for _, v in h12]
    sum12 = sum(vals12, ZERO)
    mean12 = mean_monthly_per_share(ticker, announcements, received, txns, today, 12)
    sd = _stdev(vals12)
    payment_mean = (sum12 / D(len(vals12))) if vals12 else None
    months_paid = len({month_key(d) for d, _ in h12})

    trend = None
    if len(vals12) >= 6:
        recent = sum(vals12[-3:], ZERO) / D(3)
        prior = sum(vals12[:-3], ZERO) / D(len(vals12) - 3)
        trend = safe_div(recent - prior, prior)

    last_value = D(last.amount_per_share) if last else (hist[-1][1] if hist else None)
    return {
        "ticker": ticker,
        "last": None
        if not last
        else {
            "amount_per_share": D(last.amount_per_share),
            "com_date": last.com_date,
            "ex_date": last.ex_date,
            "payment_date": last.payment_date,
            "kind": last.kind,
            "source": last.source,
        },
        "history_12m": [{"payment_date": d, "amount_per_share": v} for d, v in h12],
        "history_24m": [{"payment_date": d, "amount_per_share": v} for d, v in h24],
        "sum_12m": sum12,
        "mean_monthly_12m": mean12,
        "stdev_12m": sd,
        "coefficient_of_variation": safe_div(sd, payment_mean) if payment_mean else None,
        "regularity_12m": D(months_paid) / TWELVE,
        "trend_last3_vs_prior": trend,
        "dy_monthly_last": safe_div(last_value, price) if (last_value is not None and price) else None,
        "dy_12m": safe_div(sum12, price) if (price and vals12) else None,
        "dy_last_annualized": safe_div(last_value * TWELVE, price) if (last_value is not None and price) else None,
        "yoc_12m": safe_div(sum12, avg_price) if (avg_price and vals12) else None,
        "data_points": len(hist),
    }


def received_txns_from(txns: list[TxnInput], announcement_links: dict[int, int | None]) -> list[DividendTxn]:
    out = []
    for t in txns:
        if t.type == TxnType.DIVIDEND and t.status == TxnStatus.CONFIRMED:
            out.append(
                DividendTxn(
                    id=t.id,
                    ticker=t.ticker,
                    payment_date=t.trade_date,
                    amount=D(t.total_amount),
                    quantity=t.quantity,
                    announcement_id=announcement_links.get(t.id),
                )
            )
    return out
