from enum import StrEnum


class TxnType(StrEnum):
    OPENING_BALANCE = "OPENING_BALANCE"  # saldo inicial (cadastro da carteira existente)
    BUY = "BUY"
    SELL = "SELL"
    SUBSCRIPTION = "SUBSCRIPTION"  # subscrição em oferta pública
    PREEMPTIVE_RIGHT = "PREEMPTIVE_RIGHT"  # exercício de direito de preferência
    SUBSCRIPTION_LEFTOVER = "SUBSCRIPTION_LEFTOVER"  # sobras de subscrição
    AMORTIZATION = "AMORTIZATION"
    SPLIT = "SPLIT"  # desdobramento
    REVERSE_SPLIT = "REVERSE_SPLIT"  # grupamento
    DIVIDEND = "DIVIDEND"  # rendimento recebido
    FEE = "FEE"  # taxa/custo avulso (custódia, corretagem não vinculada a operação)
    CASH_DEPOSIT = "CASH_DEPOSIT"
    CASH_WITHDRAWAL = "CASH_WITHDRAWAL"


class TxnStatus(StrEnum):
    CONFIRMED = "CONFIRMED"  # liquidado e refletido na posição
    PENDING = "PENDING"  # executado/pago, aguardando liquidação ou conversão (ex.: recibo de subscrição)
    PLANNED = "PLANNED"  # ordem planejada, não executada — não afeta posição nem caixa


class AssetClass(StrEnum):
    FII = "FII"
    STOCK = "STOCK"
    ETF = "ETF"
    TREASURY = "TREASURY"
    FIXED_INCOME = "FIXED_INCOME"
    RIGHT = "RIGHT"  # direito de subscrição negociável (ex.: KNCR12)


class DividendKind(StrEnum):
    RENDIMENTO = "RENDIMENTO"
    AMORTIZACAO = "AMORTIZACAO"


class InvestorProfile(StrEnum):
    CONSERVADOR = "CONSERVADOR"
    EQUILIBRADO = "EQUILIBRADO"
    RENDA = "RENDA"


ACQUISITION_TYPES = {
    TxnType.OPENING_BALANCE,
    TxnType.BUY,
    TxnType.SUBSCRIPTION,
    TxnType.PREEMPTIVE_RIGHT,
    TxnType.SUBSCRIPTION_LEFTOVER,
}
SPLIT_TYPES = {TxnType.SPLIT, TxnType.REVERSE_SPLIT}
ASSET_REQUIRED_TYPES = ACQUISITION_TYPES | SPLIT_TYPES | {TxnType.SELL, TxnType.AMORTIZATION, TxnType.DIVIDEND}
CASH_ONLY_TYPES = {TxnType.CASH_DEPOSIT, TxnType.CASH_WITHDRAWAL}
