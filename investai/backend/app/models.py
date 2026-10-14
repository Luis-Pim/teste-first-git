from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import JSON, Boolean, Date, DateTime, Enum, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.domain.enums import AssetClass, DividendKind, InvestorProfile, TxnStatus, TxnType

MONEY = Numeric(20, 2)
QTY = Numeric(20, 6)
PRICE = Numeric(20, 8)
PCT = Numeric(10, 6)


def _enum(e):
    return Enum(e, native_enum=False, length=32, validate_strings=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Portfolio(Base):
    __tablename__ = "portfolios"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120))
    base_currency: Mapped[str] = mapped_column(String(3), default="BRL")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(200))
    asset_class: Mapped[AssetClass] = mapped_column(_enum(AssetClass), default=AssetClass.FII)
    segment: Mapped[str | None] = mapped_column(String(60))
    manager: Mapped[str | None] = mapped_column(String(120))
    cnpj: Mapped[str | None] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ImportBatch(Base):
    __tablename__ = "import_batches"
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255))
    fmt: Mapped[str] = mapped_column(String(40))
    rows_total: Mapped[int] = mapped_column(Integer, default=0)
    rows_imported: Mapped[int] = mapped_column(Integer, default=0)
    rows_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    rows_error: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (UniqueConstraint("portfolio_id", "external_hash", name="uq_txn_portfolio_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), index=True)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("assets.id", ondelete="RESTRICT"), index=True)
    type: Mapped[TxnType] = mapped_column(_enum(TxnType))
    status: Mapped[TxnStatus] = mapped_column(_enum(TxnStatus), default=TxnStatus.CONFIRMED)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    settlement_date: Mapped[date | None] = mapped_column(Date)
    quantity: Mapped[Decimal | None] = mapped_column(QTY)
    unit_price: Mapped[Decimal | None] = mapped_column(PRICE)
    fees: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    total_amount: Mapped[Decimal | None] = mapped_column(MONEY)
    ratio_from: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    ratio_to: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    announcement_id: Mapped[int | None] = mapped_column(ForeignKey("dividend_announcements.id", ondelete="SET NULL"))
    source: Mapped[str] = mapped_column(String(40), default="MANUAL")
    external_hash: Mapped[str | None] = mapped_column(String(64))
    import_batch_id: Mapped[int | None] = mapped_column(ForeignKey("import_batches.id", ondelete="SET NULL"))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    asset: Mapped[Asset | None] = relationship(lazy="joined")


class PriceQuote(Base):
    __tablename__ = "price_quotes"
    __table_args__ = (UniqueConstraint("asset_id", "quote_date", "source", name="uq_quote_asset_date_source"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    quote_date: Mapped[date] = mapped_column(Date, index=True)
    close: Mapped[Decimal] = mapped_column(PRICE)
    volume: Mapped[Decimal | None] = mapped_column(Numeric(24, 2))  # cotas negociadas no dia (liquidez)
    source: Mapped[str] = mapped_column(String(40))  # MANUAL, BRAPI, B3, DEMO...
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DividendAnnouncement(Base):
    __tablename__ = "dividend_announcements"
    __table_args__ = (UniqueConstraint("asset_id", "com_date", "kind", "source", name="uq_div_asset_com_kind_source"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    kind: Mapped[DividendKind] = mapped_column(_enum(DividendKind), default=DividendKind.RENDIMENTO)
    amount_per_share: Mapped[Decimal] = mapped_column(PRICE)
    com_date: Mapped[date] = mapped_column(Date)
    ex_date: Mapped[date | None] = mapped_column(Date)
    payment_date: Mapped[date] = mapped_column(Date, index=True)
    source: Mapped[str] = mapped_column(String(40), default="MANUAL")
    source_url: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    asset: Mapped[Asset] = relationship(lazy="joined")


class FundIndicator(Base):
    """Indicadores fundamentalistas por data de referência, sempre com fonte (CVM, relatório gerencial, manual)."""

    __tablename__ = "fund_indicators"
    __table_args__ = (UniqueConstraint("asset_id", "ref_date", "source", name="uq_indicator_asset_date_source"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    ref_date: Mapped[date] = mapped_column(Date, index=True)
    vp_per_share: Mapped[Decimal | None] = mapped_column(PRICE)
    vacancy: Mapped[Decimal | None] = mapped_column(PCT)
    default_rate: Mapped[Decimal | None] = mapped_column(PCT)
    leverage: Mapped[Decimal | None] = mapped_column(PCT)
    source: Mapped[str] = mapped_column(String(40), default="MANUAL")
    source_url: Mapped[str | None] = mapped_column(String(500))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    asset: Mapped[Asset] = relationship(lazy="joined")


class FundUniverse(Base):
    """Todos os fundos listados (BRAPI) enriquecidos com o informe da CVM. Base da triagem de fundos novos."""

    __tablename__ = "fund_universe"
    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(200))
    sub_type: Mapped[str] = mapped_column(String(16))  # fii, fi-agro, fi-infra
    close: Mapped[Decimal | None] = mapped_column(PRICE)
    volume: Mapped[Decimal | None] = mapped_column(Numeric(24, 2))
    listed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # quando a BRAPI foi lida
    cnpj: Mapped[str | None] = mapped_column(String(20))
    segment: Mapped[str | None] = mapped_column(String(80))
    mandate: Mapped[str | None] = mapped_column(String(80))
    net_worth: Mapped[Decimal | None] = mapped_column(Numeric(24, 2))  # patrimônio líquido
    shareholders: Mapped[int | None] = mapped_column(Integer)
    vp_per_share: Mapped[Decimal | None] = mapped_column(PRICE)
    cvm_ref_date: Mapped[date | None] = mapped_column(Date)


class StrategyConfig(Base):
    __tablename__ = "strategy_configs"
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), unique=True)
    monthly_contribution: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("300"))
    reinvest_pct: Mapped[Decimal] = mapped_column(PCT, default=Decimal("1"))
    annual_increase_pct: Mapped[Decimal] = mapped_column(PCT, default=Decimal("0"))
    extra_contributions: Mapped[list] = mapped_column(JSON, default=list)  # [{month, amount, label}]
    start_month: Mapped[date] = mapped_column(Date)
    initial_cash: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    profile: Mapped[InvestorProfile] = mapped_column(_enum(InvestorProfile), default=InvestorProfile.EQUILIBRADO)
    concentration_limit_asset: Mapped[Decimal] = mapped_column(PCT, default=Decimal("0.25"))
    concentration_limit_segment: Mapped[Decimal] = mapped_column(PCT, default=Decimal("0.40"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BenchmarkPoint(Base):
    """Séries de referência (CDI, IPCA, IFIX). Preenchidas pelos adaptadores na Fase 2."""

    __tablename__ = "benchmark_points"
    __table_args__ = (UniqueConstraint("code", "ref_date", name="uq_benchmark_code_date"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(20), index=True)
    ref_date: Mapped[date] = mapped_column(Date)
    value: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    source: Mapped[str] = mapped_column(String(40))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DataFetchLog(Base):
    """Rastreabilidade: toda chamada a fonte externa fica registrada."""

    __tablename__ = "data_fetch_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(40))
    operation: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20))  # OK, ERROR, SKIPPED
    detail: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
