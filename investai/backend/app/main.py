from decimal import Decimal

from fastapi import FastAPI
from fastapi.encoders import ENCODERS_BY_TYPE
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import assistant, market, planning, portfolio, recommendations, universe


def _decimal_to_str(d: Decimal) -> str:
    # Valores monetários trafegam como string: o encoder padrão do FastAPI converteria para float.
    if d.as_tuple().exponent < -10:
        d = d.quantize(Decimal("1e-10"))
    return format(d, "f")


ENCODERS_BY_TYPE[Decimal] = _decimal_to_str

app = FastAPI(
    title="InvestAI API",
    version="0.1.0",
    description="Acompanhamento, análise e planejamento de investimentos (FIIs). Não executa ordens reais.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(portfolio.router)
app.include_router(market.router)
app.include_router(planning.router)
app.include_router(recommendations.router)
app.include_router(universe.router)
app.include_router(assistant.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
