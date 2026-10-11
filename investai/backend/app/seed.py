"""Carga inicial da carteira do usuário.

Uso:
  python -m app.seed               # carteira real informada (quantidades), sem preços inventados
  python -m app.seed --demo-market # + cotações e proventos DEMONSTRATIVOS (fonte "DEMO") para explorar a interface
  python -m app.seed --clear-demo  # remove todos os dados com fonte "DEMO"

Preços médios NÃO foram informados: os saldos iniciais entram sem custo (cost_known = False) até
o usuário editá-los em Movimentações. Nada é estimado silenciosamente.
"""

import argparse
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import delete, select

from app.core.db import SessionLocal
from app.domain.dividends import add_months
from app.domain.enums import AssetClass, TxnStatus, TxnType
from app.models import Asset, DividendAnnouncement, PriceQuote, StrategyConfig, Transaction
from app.services.portfolio_service import DEFAULT_PORTFOLIO_ID, ensure_portfolio

ASSETS = [
    ("KNCR11", "Kinea Rendimentos Imobiliários", "Recebíveis (CRI)", "Kinea"),
    ("KNRI11", "Kinea Renda Imobiliária", "Híbrido", "Kinea"),
    ("MXRF11", "Maxi Renda", "Recebíveis (CRI)", "XP Asset"),
    ("RURA11", "Itaú Asset Rural Fiagro", "Agro (Fiagro)", "Itaú Asset"),
    ("SNEL11", "Suno Energias Limpas", "Energia", "Suno Asset"),
    ("VILG11", "Vinci Logística", "Logística", "Vinci Partners"),
    ("XPML11", "XP Malls", "Shoppings", "XP Asset"),
]
POSITIONS = {"KNCR11": 17, "KNRI11": 8, "MXRF11": 50, "RURA11": 26, "SNEL11": 24, "VILG11": 5, "XPML11": 17}
OPENING_DATE = date(2026, 9, 30)

# Valores ILUSTRATIVOS para demonstração da interface. NÃO são cotações nem proventos reais.
DEMO = {
    "KNCR11": (Decimal("105.40"), Decimal("1.05")),
    "KNRI11": (Decimal("152.10"), Decimal("1.00")),
    "MXRF11": (Decimal("9.62"), Decimal("0.10")),
    "RURA11": (Decimal("9.05"), Decimal("0.11")),
    "SNEL11": (Decimal("8.70"), Decimal("0.12")),
    "VILG11": (Decimal("92.30"), Decimal("0.70")),
    "XPML11": (Decimal("106.80"), Decimal("0.92")),
}


def seed_portfolio(db) -> bool:
    ensure_portfolio(db, DEFAULT_PORTFOLIO_ID)
    if db.scalar(select(Transaction).limit(1)):
        print("Carteira já possui operações — seed ignorado.")
        return False
    assets = {}
    for ticker, name, segment, manager in ASSETS:
        a = db.scalar(select(Asset).where(Asset.ticker == ticker)) or Asset(ticker=ticker)
        a.name, a.segment, a.manager, a.asset_class = name, segment, manager, AssetClass.FII
        a.notes = "Cadastro inicial — confira nome/segmento."
        db.add(a)
        assets[ticker] = a
    db.flush()
    for ticker, qty in POSITIONS.items():
        db.add(Transaction(
            portfolio_id=DEFAULT_PORTFOLIO_ID, asset_id=assets[ticker].id, type=TxnType.OPENING_BALANCE,
            status=TxnStatus.CONFIRMED, trade_date=OPENING_DATE, quantity=Decimal(qty), unit_price=None, source="SEED",
            notes="Saldo inicial informado pelo usuário. Preço médio não informado — edite para habilitar P/L e YoC.",
        ))
    db.add(Transaction(
        portfolio_id=DEFAULT_PORTFOLIO_ID, asset_id=assets["KNCR11"].id, type=TxnType.PREEMPTIVE_RIGHT,
        status=TxnStatus.PENDING, trade_date=OPENING_DATE, quantity=Decimal(10), total_amount=Decimal("1046.80"),
        source="SEED", notes="Exercício de direito — 13ª emissão. Valor aproximado; aguardando liquidação e conversão. Data a confirmar.",
    ))
    db.add(Transaction(
        portfolio_id=DEFAULT_PORTFOLIO_ID, asset_id=assets["MXRF11"].id, type=TxnType.BUY,
        status=TxnStatus.PLANNED, trade_date=date(2026, 10, 31), quantity=Decimal(100),
        source="SEED", notes="Compra planejada, ainda não executada.",
    ))
    if not db.scalar(select(StrategyConfig).where(StrategyConfig.portfolio_id == DEFAULT_PORTFOLIO_ID)):
        db.add(StrategyConfig(
            portfolio_id=DEFAULT_PORTFOLIO_ID, monthly_contribution=Decimal("300"), reinvest_pct=Decimal("1"),
            annual_increase_pct=Decimal("0"), start_month=date(2026, 10, 1), initial_cash=Decimal("0"),
            extra_contributions=[
                {"month": 6, "amount": "0", "label": "Extra 1 — defina o valor"},
                {"month": 12, "amount": "0", "label": "Extra 2 — defina o valor"},
            ],
        ))
    db.commit()
    print("Carteira inicial cadastrada.")
    return True


def seed_demo_market(db, today: date) -> None:
    for ticker, (price, div) in DEMO.items():
        a = db.scalar(select(Asset).where(Asset.ticker == ticker))
        if a is None:
            continue
        if not db.scalar(select(PriceQuote).where(PriceQuote.asset_id == a.id, PriceQuote.source == "DEMO")):
            db.add(PriceQuote(asset_id=a.id, quote_date=today - timedelta(days=1), close=price, source="DEMO"))
        first = date(today.year, today.month, 1)
        for i in range(-11, 1):
            m = add_months(first, i)
            com = date(m.year, m.month, 1) - timedelta(days=1)  # último dia útil aproximado do mês anterior
            pay = date(m.year, m.month, 14)
            exists = db.scalar(select(DividendAnnouncement).where(
                DividendAnnouncement.asset_id == a.id, DividendAnnouncement.com_date == com, DividendAnnouncement.source == "DEMO"))
            if not exists:
                db.add(DividendAnnouncement(asset_id=a.id, amount_per_share=div, com_date=com, ex_date=com + timedelta(days=1),
                                            payment_date=pay, source="DEMO"))
    db.commit()
    print("Dados DEMONSTRATIVOS (fonte DEMO) inseridos — não são dados reais de mercado.")


def clear_demo(db) -> None:
    db.execute(delete(PriceQuote).where(PriceQuote.source == "DEMO"))
    db.execute(delete(DividendAnnouncement).where(DividendAnnouncement.source == "DEMO"))
    db.commit()
    print("Dados DEMO removidos.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-market", action="store_true")
    ap.add_argument("--clear-demo", action="store_true")
    args = ap.parse_args()
    from app.services.portfolio_service import today_br

    with SessionLocal() as db:
        if args.clear_demo:
            clear_demo(db)
        else:
            seed_portfolio(db)
            if args.demo_market:
                seed_demo_market(db, today_br())
