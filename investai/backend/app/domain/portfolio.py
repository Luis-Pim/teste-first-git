"""Motor de posições: reconstrói saldo de cotas, custo médio e fluxo de caixa a partir das operações.

Regras (mercado brasileiro, FIIs):
- A quantidade de cotas vem SEMPRE do replay das operações, nunca de saldo/preço médio.
- Custo médio ponderado: compras/subscrições somam (qtd × preço + taxas) ao custo; vendas baixam
  custo pelo preço médio vigente (não alteram o preço médio) e geram lucro/prejuízo realizado.
- Amortização devolve capital: reduz o custo total (e portanto o preço médio), sem mudar a quantidade.
- Desdobramento/grupamento multiplicam a quantidade por ratio_to/ratio_from; o custo total não muda.
- Operações PENDING (ex.: recibo de subscrição) ficam em balde separado até a confirmação.
- Operações PLANNED não afetam posição nem caixa.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from app.domain.enums import ACQUISITION_TYPES, SPLIT_TYPES, TxnStatus, TxnType
from app.domain.money import ZERO, D, money, safe_div


class PositionError(ValueError):
    pass


def _fmt(q) -> str:
    return format(D(q).normalize(), "f")


class OversellError(PositionError):
    def __init__(self, ticker: str, on: date, held, attempted):
        self.ticker, self.on, self.held, self.attempted = ticker, on, held, attempted
        super().__init__(
            f"Venda de {_fmt(attempted)} cotas de {ticker} em {on.strftime('%d/%m/%Y')} excede o saldo de {_fmt(held)} cotas."
        )


@dataclass(frozen=True)
class TxnInput:
    """Representação pura de uma operação (desacoplada do ORM)."""

    type: TxnType
    trade_date: date
    status: TxnStatus = TxnStatus.CONFIRMED
    ticker: str | None = None
    quantity: object = None
    unit_price: object = None
    fees: object = ZERO
    total_amount: object = None
    ratio_from: object = None
    ratio_to: object = None
    id: int = 0

    def gross_cost(self):
        """Custo de aquisição (qtd × preço + taxas) ou None se o preço é desconhecido."""
        if self.total_amount is not None and self.type != TxnType.SELL:
            return D(self.total_amount)
        if self.unit_price is None or self.quantity is None:
            return None
        return D(self.quantity) * D(self.unit_price) + D(self.fees)

    def sell_proceeds(self):
        if self.total_amount is not None:
            return D(self.total_amount)
        return D(self.quantity) * D(self.unit_price) - D(self.fees)


# Ordem de processamento dentro do mesmo dia: eventos corporativos antes de negociações,
# compras antes de vendas (evita falso "vendeu sem ter" em day trade), caixa por último.
_TYPE_ORDER = {
    TxnType.OPENING_BALANCE: 0,
    TxnType.SPLIT: 1,
    TxnType.REVERSE_SPLIT: 1,
    TxnType.CASH_DEPOSIT: 2,
    TxnType.BUY: 3,
    TxnType.SUBSCRIPTION: 3,
    TxnType.PREEMPTIVE_RIGHT: 3,
    TxnType.SUBSCRIPTION_LEFTOVER: 3,
    TxnType.SELL: 4,
    TxnType.AMORTIZATION: 5,
    TxnType.DIVIDEND: 6,
    TxnType.FEE: 7,
    TxnType.CASH_WITHDRAWAL: 8,
}


def sort_key(t: TxnInput):
    return (t.trade_date, _TYPE_ORDER[t.type], t.id)


@dataclass
class Position:
    ticker: str
    quantity: object = ZERO
    total_cost: object = ZERO
    cost_known: bool = True
    pending_quantity: object = ZERO
    pending_cost: object = ZERO
    pending_cost_known: bool = True
    planned_quantity: object = ZERO
    planned_cost: object = ZERO
    realized_pnl: object = ZERO
    dividends_received: object = ZERO
    amortizations_received: object = ZERO
    first_date: date | None = None

    @property
    def avg_price(self):
        if not self.cost_known or self.quantity <= 0:
            return None
        return self.total_cost / self.quantity

    @property
    def is_open(self) -> bool:
        return self.quantity > 0 or self.pending_quantity > 0 or self.planned_quantity > 0


@dataclass
class CashFlows:
    deposits: object = ZERO
    withdrawals: object = ZERO
    purchases: object = ZERO  # caixa gasto em compras/subscrições (inclui PENDING já pagas)
    sales: object = ZERO
    dividends: object = ZERO
    amortizations: object = ZERO
    fees: object = ZERO  # taxas avulsas

    @property
    def balance(self):
        return (
            self.deposits - self.withdrawals - self.purchases + self.sales + self.dividends + self.amortizations - self.fees
        )


@dataclass
class PortfolioState:
    positions: dict[str, Position] = field(default_factory=dict)
    cash: CashFlows = field(default_factory=CashFlows)
    warnings: list[str] = field(default_factory=list)

    def position(self, ticker: str) -> Position:
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker=ticker)
        return self.positions[ticker]


def build_portfolio(txns: Iterable[TxnInput], as_of: date | None = None, include_pending: bool = True) -> PortfolioState:
    state = PortfolioState()
    for t in sorted(txns, key=sort_key):
        if as_of is not None and t.trade_date > as_of:
            continue
        if t.status == TxnStatus.PENDING and not include_pending:
            continue
        _apply(state, t)
    return state


def _require_ticker(t: TxnInput) -> str:
    if not t.ticker:
        raise PositionError(f"Operação {t.type} exige um ativo.")
    return t.ticker


def _apply(state: PortfolioState, t: TxnInput) -> None:
    cash = state.cash

    if t.status == TxnStatus.PLANNED:
        if t.type in ACQUISITION_TYPES:
            p = state.position(_require_ticker(t))
            p.planned_quantity += D(t.quantity)
            cost = t.gross_cost()
            if cost is not None:
                p.planned_cost += cost
        return

    if t.type in ACQUISITION_TYPES:
        p = state.position(_require_ticker(t))
        qty = D(t.quantity)
        if qty <= 0:
            raise PositionError("Quantidade deve ser positiva.")
        cost = t.gross_cost()
        if t.status == TxnStatus.PENDING:
            p.pending_quantity += qty
            if cost is None:
                p.pending_cost_known = False
            else:
                p.pending_cost += cost
                cash.purchases += cost
            return
        p.quantity += qty
        p.first_date = p.first_date or t.trade_date
        if cost is None:
            p.cost_known = False
            state.warnings.append(f"{p.ticker}: preço de aquisição não informado em {t.trade_date.isoformat()}.")
        else:
            p.total_cost += cost
            if t.type != TxnType.OPENING_BALANCE:
                cash.purchases += cost
        return

    if t.type == TxnType.SELL:
        if t.status == TxnStatus.PENDING:
            return  # venda só baixa a posição quando liquidada
        p = state.position(_require_ticker(t))
        qty = D(t.quantity)
        if qty > p.quantity:
            raise OversellError(p.ticker, t.trade_date, p.quantity, qty)
        proceeds = t.sell_proceeds()
        if p.cost_known and p.quantity > 0:
            cost_out = p.total_cost * qty / p.quantity
            p.realized_pnl += proceeds - cost_out
            p.total_cost -= cost_out
        p.quantity -= qty
        if p.quantity == 0:
            p.total_cost = ZERO
            p.cost_known = True
        cash.sales += proceeds
        return

    if t.type in SPLIT_TYPES:
        p = state.position(_require_ticker(t))
        r_from, r_to = D(t.ratio_from), D(t.ratio_to)
        if r_from <= 0 or r_to <= 0:
            raise PositionError("Proporção de desdobramento/grupamento inválida.")
        factor = r_to / r_from
        p.quantity *= factor
        p.pending_quantity *= factor
        return

    if t.type == TxnType.AMORTIZATION:
        p = state.position(_require_ticker(t))
        amount = D(t.total_amount)
        if amount == 0 and t.unit_price is not None:
            amount = D(t.unit_price) * (D(t.quantity) if t.quantity is not None else p.quantity)
        p.amortizations_received += amount
        if p.cost_known:
            p.total_cost = max(ZERO, p.total_cost - amount)
        cash.amortizations += amount
        return

    if t.type == TxnType.DIVIDEND:
        p = state.position(_require_ticker(t))
        amount = D(t.total_amount)
        p.dividends_received += amount
        cash.dividends += amount
        return

    if t.type == TxnType.FEE:
        cash.fees += D(t.total_amount)
        return
    if t.type == TxnType.CASH_DEPOSIT:
        cash.deposits += D(t.total_amount)
        return
    if t.type == TxnType.CASH_WITHDRAWAL:
        cash.withdrawals += D(t.total_amount)
        return

    raise PositionError(f"Tipo de operação não suportado: {t.type}")


def quantity_on(txns: Iterable[TxnInput], ticker: str, on: date) -> object:
    """Cotas confirmadas detidas ao final de `on` (usado para elegibilidade na data-com)."""
    confirmed = [t for t in txns if t.status == TxnStatus.CONFIRMED and t.ticker == ticker]
    state = build_portfolio(confirmed, as_of=on, include_pending=False)
    p = state.positions.get(ticker)
    return p.quantity if p else ZERO


def unrealized(position: Position, price) -> dict:
    """Valor de mercado e resultado não realizado. Não confunde custo com valor de mercado."""
    if price is None:
        return {"market_value": None, "unrealized_pnl": None, "unrealized_pct": None}
    mv = money(position.quantity * D(price))
    if not position.cost_known:
        return {"market_value": mv, "unrealized_pnl": None, "unrealized_pct": None}
    pnl = mv - money(position.total_cost)
    pct = safe_div(pnl, position.total_cost)
    return {"market_value": mv, "unrealized_pnl": pnl, "unrealized_pct": pct}
