from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.enums import (
    ACQUISITION_TYPES,
    ASSET_REQUIRED_TYPES,
    SPLIT_TYPES,
    AssetClass,
    DividendKind,
    InvestorProfile,
    TxnStatus,
    TxnType,
)


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------- ativos ----------
class AssetIn(BaseModel):
    ticker: str = Field(min_length=4, max_length=16)
    name: str | None = None
    asset_class: AssetClass = AssetClass.FII
    segment: str | None = None
    manager: str | None = None
    cnpj: str | None = None
    notes: str | None = None

    @field_validator("ticker")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.strip().upper()


class AssetPatch(BaseModel):
    name: str | None = None
    asset_class: AssetClass | None = None
    segment: str | None = None
    manager: str | None = None
    cnpj: str | None = None
    notes: str | None = None
    is_active: bool | None = None


class AssetOut(ORM):
    id: int
    ticker: str
    name: str | None
    asset_class: AssetClass
    segment: str | None
    manager: str | None
    cnpj: str | None
    is_active: bool
    notes: str | None


# ---------- operações ----------
class TransactionIn(BaseModel):
    type: TxnType
    status: TxnStatus = TxnStatus.CONFIRMED
    trade_date: date
    settlement_date: date | None = None
    ticker: str | None = None
    quantity: Decimal | None = Field(default=None, gt=0)
    unit_price: Decimal | None = Field(default=None, ge=0)
    fees: Decimal = Field(default=Decimal("0"), ge=0)
    total_amount: Decimal | None = Field(default=None, ge=0)
    ratio_from: Decimal | None = Field(default=None, gt=0)
    ratio_to: Decimal | None = Field(default=None, gt=0)
    announcement_id: int | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def check(self):
        t = self.type
        if self.ticker:
            self.ticker = self.ticker.strip().upper()
        if t in ASSET_REQUIRED_TYPES and not self.ticker:
            raise ValueError("Esta operação exige um ativo (ticker).")
        if t in ACQUISITION_TYPES | {TxnType.SELL} and self.quantity is None:
            raise ValueError("Informe a quantidade de cotas.")
        if t == TxnType.SELL and self.unit_price is None and self.total_amount is None:
            raise ValueError("Venda exige preço unitário ou valor total.")
        if t in SPLIT_TYPES and (self.ratio_from is None or self.ratio_to is None):
            raise ValueError("Informe a proporção (de → para).")
        if t in {TxnType.DIVIDEND, TxnType.AMORTIZATION, TxnType.FEE, TxnType.CASH_DEPOSIT, TxnType.CASH_WITHDRAWAL}:
            if self.total_amount is None:
                raise ValueError("Informe o valor total.")
        if t in ACQUISITION_TYPES and self.status != TxnStatus.PLANNED and t != TxnType.OPENING_BALANCE:
            if self.unit_price is None and self.total_amount is None:
                raise ValueError("Compra/subscrição executada exige preço unitário ou valor total.")
        return self


class TransactionOut(ORM):
    id: int
    type: TxnType
    status: TxnStatus
    trade_date: date
    settlement_date: date | None
    ticker: str | None = None
    quantity: Decimal | None
    unit_price: Decimal | None
    fees: Decimal
    total_amount: Decimal | None
    ratio_from: Decimal | None
    ratio_to: Decimal | None
    announcement_id: int | None
    source: str
    notes: str | None

    @model_validator(mode="before")
    @classmethod
    def ticker_from_asset(cls, data):
        if hasattr(data, "asset"):
            obj = {c: getattr(data, c) for c in cls.model_fields if c != "ticker" and hasattr(data, c)}
            obj["ticker"] = data.asset.ticker if data.asset else None
            return obj
        return data


class ConfirmIn(BaseModel):
    settlement_date: date | None = None
    trade_date: date | None = None
    unit_price: Decimal | None = Field(default=None, ge=0)
    total_amount: Decimal | None = Field(default=None, ge=0)


# ---------- cotações e proventos ----------
class QuoteIn(BaseModel):
    ticker: str
    quote_date: date
    close: Decimal = Field(gt=0)
    source: str = "MANUAL"


class QuoteOut(BaseModel):
    ticker: str
    quote_date: date
    close: Decimal
    source: str


class AnnouncementIn(BaseModel):
    ticker: str
    kind: DividendKind = DividendKind.RENDIMENTO
    amount_per_share: Decimal = Field(gt=0)
    com_date: date
    ex_date: date | None = None
    payment_date: date
    source: str = "MANUAL"
    source_url: str | None = None

    @model_validator(mode="after")
    def dates(self):
        if self.payment_date < self.com_date:
            raise ValueError("Data de pagamento não pode ser anterior à data-com.")
        return self


class AnnouncementOut(BaseModel):
    id: int
    ticker: str
    kind: DividendKind
    amount_per_share: Decimal
    com_date: date
    ex_date: date | None
    payment_date: date
    source: str
    source_url: str | None


# ---------- estratégia ----------
class ExtraIn(BaseModel):
    month: int = Field(ge=1, le=12)
    amount: Decimal = Field(ge=0)
    label: str = ""


class StrategyIn(BaseModel):
    monthly_contribution: Decimal = Field(ge=0)
    reinvest_pct: Decimal = Field(ge=0, le=1)
    annual_increase_pct: Decimal = Field(ge=0, le=1)
    extra_contributions: list[ExtraIn] = []
    start_month: date
    initial_cash: Decimal = Decimal("0")
    profile: InvestorProfile = InvestorProfile.EQUILIBRADO
    concentration_limit_asset: Decimal = Field(default=Decimal("0.25"), gt=0, le=1)
    concentration_limit_segment: Decimal = Field(default=Decimal("0.40"), gt=0, le=1)


class StrategyOut(StrategyIn, ORM):
    pass


# ---------- simulador ----------
class AssumptionsIn(BaseModel):
    dividend_yield_annual: Decimal = Field(ge=0, le=Decimal("0.5"))
    dividend_growth_annual: Decimal = Field(ge=-1, le=1)
    price_growth_annual: Decimal = Field(ge=-1, le=1)
    inflation_annual: Decimal = Field(ge=-1, le=1)


class SimulationIn(BaseModel):
    initial_value: Decimal | None = Field(default=None, ge=0, description="Vazio = valor atual da carteira")
    monthly_contribution: Decimal = Field(default=Decimal("300"), ge=0)
    annual_contribution_increase: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    extras: list[ExtraIn] = []
    reinvest_pct: Decimal = Field(default=Decimal("1"), ge=0, le=1)
    years: int = Field(default=10, ge=1, le=30)
    scenario: str = "base"
    assumptions: AssumptionsIn | None = None  # sobrescreve o cenário escolhido
    scenario_overrides: dict[str, AssumptionsIn] | None = None


class GoalTimeIn(SimulationIn):
    target_monthly_income: Decimal = Field(gt=0)
    real: bool = False


class GoalContributionIn(SimulationIn):
    target_monthly_income: Decimal = Field(gt=0)
    real: bool = False


class ExtrasImpactIn(SimulationIn):
    compare_extras: list[ExtraIn]
