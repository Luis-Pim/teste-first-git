from datetime import date
from decimal import Decimal as Dec

import pytest

from app.domain.enums import TxnStatus as S
from app.domain.enums import TxnType as T
from app.domain.portfolio import OversellError, TxnInput, build_portfolio, quantity_on, unrealized


def tx(type, d, ticker="KNRI11", **kw):
    return TxnInput(type=type, trade_date=d, ticker=ticker, **kw)


def test_weighted_average_price_includes_fees():
    txns = [
        tx(T.BUY, date(2026, 1, 10), quantity=10, unit_price=Dec("100"), fees=Dec("2")),
        tx(T.BUY, date(2026, 2, 10), quantity=5, unit_price=Dec("130"), fees=Dec("1")),
    ]
    p = build_portfolio(txns).positions["KNRI11"]
    assert p.quantity == 15
    assert p.total_cost == Dec("1653")  # 1002 + 651
    assert p.avg_price == Dec("1653") / 15


def test_partial_sell_keeps_avg_price_and_realizes_pnl():
    txns = [
        tx(T.BUY, date(2026, 1, 10), quantity=10, unit_price=Dec("100")),
        tx(T.SELL, date(2026, 3, 1), quantity=4, unit_price=Dec("110"), fees=Dec("1")),
    ]
    st = build_portfolio(txns)
    p = st.positions["KNRI11"]
    assert p.quantity == 6
    assert p.avg_price == Dec("100")
    assert p.realized_pnl == Dec("39")  # 440 - 1 - 400
    assert st.cash.sales == Dec("439")


def test_full_sell_resets_cost():
    txns = [tx(T.BUY, date(2026, 1, 1), quantity=3, unit_price=Dec("10")), tx(T.SELL, date(2026, 1, 2), quantity=3, unit_price=Dec("12"))]
    p = build_portfolio(txns).positions["KNRI11"]
    assert p.quantity == 0 and p.total_cost == 0 and p.avg_price is None


def test_oversell_rejected():
    with pytest.raises(OversellError):
        build_portfolio([tx(T.BUY, date(2026, 1, 1), quantity=1, unit_price=1), tx(T.SELL, date(2026, 1, 2), quantity=2, unit_price=1)])


def test_same_day_buy_then_sell_ordering():
    txns = [
        tx(T.SELL, date(2026, 1, 1), quantity=1, unit_price=Dec("11"), id=1),
        tx(T.BUY, date(2026, 1, 1), quantity=1, unit_price=Dec("10"), id=2),
    ]
    assert build_portfolio(txns).positions["KNRI11"].quantity == 0


def test_split_and_reverse_split_keep_total_cost():
    txns = [
        tx(T.BUY, date(2026, 1, 1), quantity=10, unit_price=Dec("100")),
        tx(T.SPLIT, date(2026, 2, 1), ratio_from=1, ratio_to=10),
    ]
    p = build_portfolio(txns).positions["KNRI11"]
    assert p.quantity == 100 and p.total_cost == 1000 and p.avg_price == 10
    txns.append(tx(T.REVERSE_SPLIT, date(2026, 3, 1), ratio_from=10, ratio_to=1))
    p = build_portfolio(txns).positions["KNRI11"]
    assert p.quantity == 10 and p.avg_price == 100


def test_amortization_reduces_cost_not_quantity():
    txns = [tx(T.BUY, date(2026, 1, 1), quantity=10, unit_price=Dec("100")), tx(T.AMORTIZATION, date(2026, 2, 1), total_amount=Dec("50"))]
    st = build_portfolio(txns)
    p = st.positions["KNRI11"]
    assert p.quantity == 10 and p.total_cost == 950 and p.avg_price == 95
    assert st.cash.amortizations == 50


def test_pending_subscription_is_separate_until_confirmed():
    sub = dict(quantity=10, total_amount=Dec("1046.80"))
    txns = [
        tx(T.OPENING_BALANCE, date(2026, 9, 30), ticker="KNCR11", quantity=17, unit_price=Dec("100")),
        tx(T.PREEMPTIVE_RIGHT, date(2026, 9, 30), ticker="KNCR11", status=S.PENDING, **sub),
    ]
    p = build_portfolio(txns).positions["KNCR11"]
    assert p.quantity == 17 and p.pending_quantity == 10 and p.pending_cost == Dec("1046.80")
    assert p.avg_price == 100
    confirmed = [txns[0], tx(T.PREEMPTIVE_RIGHT, date(2026, 9, 30), ticker="KNCR11", **sub)]
    p = build_portfolio(confirmed).positions["KNCR11"]
    assert p.quantity == 27 and p.total_cost == Dec("2746.80")


def test_planned_orders_do_not_affect_position_or_cash():
    txns = [
        tx(T.OPENING_BALANCE, date(2026, 9, 30), ticker="MXRF11", quantity=50, unit_price=Dec("9.5")),
        tx(T.BUY, date(2026, 10, 15), ticker="MXRF11", quantity=100, status=S.PLANNED),
    ]
    st = build_portfolio(txns)
    p = st.positions["MXRF11"]
    assert p.quantity == 50 and p.planned_quantity == 100
    assert st.cash.purchases == 0


def test_opening_balance_without_price_flags_unknown_cost():
    st = build_portfolio([tx(T.OPENING_BALANCE, date(2026, 9, 30), quantity=8)])
    p = st.positions["KNRI11"]
    assert p.quantity == 8 and p.cost_known is False and p.avg_price is None
    assert unrealized(p, Dec("150"))["unrealized_pnl"] is None
    assert unrealized(p, Dec("150"))["market_value"] == Dec("1200.00")
    assert st.warnings


def test_reinvested_dividend_and_cash_balance():
    txns = [
        TxnInput(type=T.CASH_DEPOSIT, trade_date=date(2026, 1, 1), total_amount=Dec("300")),
        tx(T.BUY, date(2026, 1, 2), quantity=2, unit_price=Dec("100")),
        tx(T.DIVIDEND, date(2026, 1, 15), total_amount=Dec("1.70")),
        tx(T.BUY, date(2026, 1, 20), quantity=1, unit_price=Dec("100")),
    ]
    st = build_portfolio(txns)
    assert st.positions["KNRI11"].quantity == 3
    assert st.positions["KNRI11"].dividends_received == Dec("1.70")
    assert st.cash.balance == Dec("1.70")


def test_quantity_on_com_date_uses_confirmed_trades_until_that_day():
    txns = [
        tx(T.BUY, date(2026, 1, 10), quantity=10, unit_price=1),
        tx(T.BUY, date(2026, 1, 30), quantity=5, unit_price=1),  # na própria data-com: elegível
        tx(T.BUY, date(2026, 1, 31), quantity=7, unit_price=1),  # após data-com: não elegível
        tx(T.BUY, date(2026, 1, 15), quantity=99, unit_price=1, status=S.PENDING),
    ]
    assert quantity_on(txns, "KNRI11", date(2026, 1, 30)) == 15
