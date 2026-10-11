"""Utilitários monetários. Todo cálculo financeiro usa Decimal — nunca float."""

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, getcontext

getcontext().prec = 34

ZERO = Decimal("0")
ONE = Decimal("1")
CENT = Decimal("0.01")
TWELVE = Decimal("12")


def D(value) -> Decimal:
    """Converte para Decimal sem passar por float binário."""
    if value is None:
        return ZERO
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(repr(value))
    return Decimal(str(value))


def money(value) -> Decimal:
    """Arredonda para centavos (meio para cima, padrão de extratos brasileiros)."""
    return D(value).quantize(CENT, rounding=ROUND_HALF_UP)


def rate(value, places: int = 6) -> Decimal:
    return D(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)


def safe_div(a, b) -> Decimal | None:
    b = D(b)
    if b == 0:
        return None
    return D(a) / b


def annual_to_monthly(annual_rate) -> Decimal:
    """Taxa anual composta -> taxa mensal equivalente: (1+a)^(1/12) - 1."""
    return (ONE + D(annual_rate)) ** (ONE / TWELVE) - ONE
