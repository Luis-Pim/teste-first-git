"""Motor de recomendação de aportes (explicável, sem promessas de rentabilidade).

Como decide:
1. Para cada fundo do universo (carteira + lista de acompanhamento) calcula notas de 0 a 100 por fator,
   cada uma com a explicação e os dados usados. Fator sem dado fica "não avaliado" e seu peso é
   redistribuído entre os fatores disponíveis; a cobertura (% do peso avaliado) é sempre informada.
2. Aplica vetos ("não comprar") que nenhuma nota alta compensa: corte relevante de rendimentos,
   prêmio excessivo sobre o valor patrimonial, liquidez muito baixa, limite de concentração,
   fundamentos ruins (vacância, inadimplência, alavancagem) e cobertura de dados insuficiente.
3. Ordena os elegíveis pela nota ponderada do perfil e distribui o orçamento respeitando os limites
   de concentração. Nunca ordena só pelo dividend yield: no perfil Renda ele pesa 30%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from app.domain.enums import InvestorProfile
from app.domain.money import ZERO, D, money, safe_div

FACTORS = ("valuation", "income", "income_quality", "liquidity", "diversification", "fundamentals", "price_position")
FACTOR_SHORT = {
    "valuation": "desconto sobre o valor patrimonial",
    "income": "dividend yield",
    "income_quality": "regularidade dos rendimentos",
    "liquidity": "liquidez",
    "diversification": "diversificação",
    "fundamentals": "fundamentos",
    "price_position": "preço frente à média",
}
FACTOR_LABEL = {
    "valuation": "Preço sobre valor patrimonial (P/VP)",
    "income": "Dividend yield 12 meses",
    "income_quality": "Regularidade e estabilidade dos rendimentos",
    "liquidity": "Liquidez diária",
    "diversification": "Efeito na diversificação",
    "fundamentals": "Vacância, inadimplência e alavancagem",
    "price_position": "Preço frente à média de 12 meses",
}

PROFILE_WEIGHTS: dict[InvestorProfile, dict[str, Decimal]] = {
    InvestorProfile.CONSERVADOR: {
        "income_quality": D("0.25"), "diversification": D("0.20"), "fundamentals": D("0.15"), "liquidity": D("0.15"),
        "valuation": D("0.15"), "income": D("0.05"), "price_position": D("0.05"),
    },
    InvestorProfile.EQUILIBRADO: {
        "valuation": D("0.20"), "income_quality": D("0.20"), "diversification": D("0.15"), "income": D("0.15"),
        "fundamentals": D("0.15"), "liquidity": D("0.10"), "price_position": D("0.05"),
    },
    InvestorProfile.RENDA: {
        "income": D("0.30"), "income_quality": D("0.25"), "valuation": D("0.15"), "fundamentals": D("0.10"),
        "liquidity": D("0.10"), "diversification": D("0.05"), "price_position": D("0.05"),
    },
}


@dataclass(frozen=True)
class Thresholds:
    """Limites dos vetos. Valores padrão conservadores, editáveis por perfil."""

    max_pvp: Decimal = D("1.10")
    dividend_cut: Decimal = D("-0.15")  # média dos últimos 3 meses vs. anteriores
    min_liquidity: Decimal = D("300000")  # R$/dia
    max_vacancy: Decimal = D("0.20")
    max_default: Decimal = D("0.05")
    max_leverage: Decimal = D("0.35")
    min_coverage: Decimal = D("0.50")
    abnormal_yield: Decimal = D("0.16")


PROFILE_THRESHOLDS = {
    InvestorProfile.CONSERVADOR: Thresholds(max_pvp=D("1.05"), min_liquidity=D("500000"), max_vacancy=D("0.15"), max_leverage=D("0.30")),
    InvestorProfile.EQUILIBRADO: Thresholds(),
    InvestorProfile.RENDA: Thresholds(max_pvp=D("1.10"), dividend_cut=D("-0.12")),
}


@dataclass
class FundMetrics:
    ticker: str
    segment: str | None
    price: Decimal | None
    held_quantity: Decimal = ZERO
    pvp: Decimal | None = None
    pvp_median_12m: Decimal | None = None
    dy_12m: Decimal | None = None
    mean_monthly_per_share: Decimal | None = None
    regularity_12m: Decimal | None = None
    dividend_cv: Decimal | None = None
    dividend_trend: Decimal | None = None
    avg_daily_liquidity: Decimal | None = None  # R$ negociados por dia
    price_vs_12m_avg: Decimal | None = None  # preço / média 12m
    vacancy: Decimal | None = None
    default_rate: Decimal | None = None
    leverage: Decimal | None = None
    is_paper: bool = False  # fundo de recebíveis: vacância não se aplica
    sources: dict[str, str] = field(default_factory=dict)  # fator -> fonte/data


@dataclass
class PortfolioContext:
    values: dict[str, Decimal]  # ticker -> valor de mercado atual
    segments: dict[str, str | None]
    limit_asset: Decimal
    limit_segment: Decimal

    @property
    def total(self) -> Decimal:
        return sum(self.values.values(), ZERO)


@dataclass
class FactorScore:
    factor: str
    score: Decimal | None  # None = não avaliado
    detail: str


def _clamp(x) -> Decimal:
    return max(ZERO, min(D(100), D(x)))


def _lerp(x, x0, y0, x1, y1) -> Decimal:
    """Interpolação linear com saturação: x0→y0, x1→y1."""
    x, x0, x1, y0, y1 = D(x), D(x0), D(x1), D(y0), D(y1)
    if x1 == x0:
        return _clamp(y1)
    t = (x - x0) / (x1 - x0)
    t = max(ZERO, min(D(1), t))
    return _clamp(y0 + (y1 - y0) * t)


def _pct(x) -> str:
    return f"{D(x) * 100:.1f}%".replace(".", ",")


def weights_after(ctx: PortfolioContext, ticker: str, segment: str | None, add_value) -> tuple[Decimal, Decimal]:
    total = ctx.total + D(add_value)
    if total == 0:
        return ZERO, ZERO
    asset = (ctx.values.get(ticker, ZERO) + D(add_value)) / total
    seg_value = sum((v for t, v in ctx.values.items() if ctx.segments.get(t) == segment), ZERO)
    seg = (seg_value + D(add_value)) / total
    return asset, seg


def score_factors(m: FundMetrics, ctx: PortfolioContext, ref_purchase) -> list[FactorScore]:
    out: list[FactorScore] = []

    # P/VP: 0,85 ou menos = 100; 1,00 = 70; 1,15 ou mais = 0. Bônus/penalidade frente à mediana própria.
    if m.pvp is not None:
        s = _lerp(m.pvp, D("0.85"), 100, D("1.00"), 70) if m.pvp <= 1 else _lerp(m.pvp, D("1.00"), 70, D("1.15"), 0)
        det = f"P/VP {m.pvp:.2f}".replace(".", ",")
        if m.pvp_median_12m:
            rel = m.pvp / m.pvp_median_12m - 1
            s = _clamp(s - rel * 100)
            det += f" ({'abaixo' if rel < 0 else 'acima'} da mediana de 12 meses, {f'{m.pvp_median_12m:.2f}'.replace('.', ',')})"
        out.append(FactorScore("valuation", s, det))
    else:
        out.append(FactorScore("valuation", None, "Sem valor patrimonial por cota (informe CVM)."))

    # DY 12m: 6% = 0; 12% = 100. Acima de 16% não ganha mais (ver alerta de yield anormal).
    if m.dy_12m is not None:
        out.append(FactorScore("income", _lerp(m.dy_12m, D("0.06"), 0, D("0.12"), 100), f"DY 12m {_pct(m.dy_12m)}"))
    else:
        out.append(FactorScore("income", None, "Sem histórico de proventos com cotação."))

    # Qualidade da renda: regularidade (50%), estabilidade (CV, 30%) e tendência (20%).
    if m.regularity_12m is not None:
        reg = _clamp(m.regularity_12m * 100)
        stab = _lerp(m.dividend_cv, D("0.05"), 100, D("0.40"), 0) if m.dividend_cv is not None else reg
        trend = _lerp(m.dividend_trend, D("-0.15"), 0, D("0.05"), 100) if m.dividend_trend is not None else D(60)
        s = reg * D("0.5") + stab * D("0.3") + trend * D("0.2")
        parts = [f"pagou em {int(m.regularity_12m * 12)} dos últimos 12 meses"]
        if m.dividend_cv is not None:
            parts.append(f"variação (CV) {_pct(m.dividend_cv)}")
        if m.dividend_trend is not None:
            parts.append(f"últimos 3 meses {'+' if m.dividend_trend >= 0 else ''}{_pct(m.dividend_trend)} vs. anteriores")
        out.append(FactorScore("income_quality", _clamp(s), "; ".join(parts)))
    else:
        out.append(FactorScore("income_quality", None, "Sem histórico de proventos."))

    # Liquidez: R$ 300 mil/dia = 0; R$ 5 milhões/dia = 100.
    if m.avg_daily_liquidity is not None:
        out.append(FactorScore("liquidity", _lerp(m.avg_daily_liquidity, D("300000"), 0, D("5000000"), 100),
                               f"R$ {money(m.avg_daily_liquidity / 1000):,.0f} mil por dia".replace(",", ".")))
    else:
        out.append(FactorScore("liquidity", None, "Sem volume negociado."))

    # Diversificação: compara o peso após a compra com o limite. Abaixo de metade do limite = 100.
    if ctx.total > 0 or ref_purchase:
        aw, sw = weights_after(ctx, m.ticker, m.segment, ref_purchase)
        a_score = _lerp(aw, ctx.limit_asset / 2, 100, ctx.limit_asset, 0)
        s_score = _lerp(sw, ctx.limit_segment / 2, 100, ctx.limit_segment, 0)
        new_segment = m.segment and not any(ctx.segments.get(t) == m.segment for t, v in ctx.values.items() if v > 0)
        s = a_score * D("0.5") + s_score * D("0.5")
        det = f"após um aporte de referência: {_pct(aw)} no fundo e {_pct(sw)} em {m.segment or 'segmento não classificado'}"
        if new_segment:
            s = _clamp(s + 10)
            det += "; adiciona um segmento novo à carteira"
        out.append(FactorScore("diversification", s, det))
    else:
        out.append(FactorScore("diversification", D(100), "Carteira vazia."))

    # Fundamentos: média das notas disponíveis.
    subs, parts = [], []
    if m.vacancy is not None and not m.is_paper:
        subs.append(_lerp(m.vacancy, D("0.02"), 100, D("0.20"), 0)); parts.append(f"vacância {_pct(m.vacancy)}")
    if m.default_rate is not None:
        subs.append(_lerp(m.default_rate, ZERO, 100, D("0.05"), 0)); parts.append(f"inadimplência {_pct(m.default_rate)}")
    if m.leverage is not None:
        subs.append(_lerp(m.leverage, D("0.05"), 100, D("0.35"), 0)); parts.append(f"alavancagem {_pct(m.leverage)}")
    if subs:
        out.append(FactorScore("fundamentals", sum(subs, ZERO) / len(subs), "; ".join(parts)))
    else:
        out.append(FactorScore("fundamentals", None, "Sem dados de vacância, inadimplência ou alavancagem (relatório gerencial)."))

    # Posição do preço: 10% abaixo da média 12m = 100; 10% acima = 0.
    if m.price_vs_12m_avg is not None:
        rel = m.price_vs_12m_avg - 1
        out.append(FactorScore("price_position", _lerp(rel, D("-0.10"), 100, D("0.10"), 0),
                               f"preço {'abaixo' if rel < 0 else 'acima'} da média de 12 meses em {_pct(abs(rel))}"))
    else:
        out.append(FactorScore("price_position", None, "Sem histórico de 12 meses de cotações."))
    return out


def vetoes_and_risks(m: FundMetrics, ctx: PortfolioContext, th: Thresholds) -> tuple[list[str], list[str]]:
    vetoes, risks = [], []
    if m.price is None or m.price <= 0:
        vetoes.append("Sem cotação: não é possível calcular quantidade nem valor.")
        return vetoes, risks
    if m.dividend_trend is not None and m.dividend_trend <= th.dividend_cut:
        vetoes.append(f"Corte relevante de rendimentos: últimos 3 meses {_pct(m.dividend_trend)} vs. período anterior.")
    if m.pvp is not None and m.pvp > th.max_pvp:
        vetoes.append(f"Prêmio elevado sobre o valor patrimonial: P/VP {m.pvp:.2f} acima de {th.max_pvp:.2f}.".replace(".", ",", 2))
    if m.avg_daily_liquidity is not None and m.avg_daily_liquidity < th.min_liquidity:
        vetoes.append(f"Liquidez baixa: menos de R$ {th.min_liquidity / 1000:.0f} mil negociados por dia.")
    if m.vacancy is not None and not m.is_paper and m.vacancy > th.max_vacancy:
        vetoes.append(f"Vacância de {_pct(m.vacancy)} acima do limite de {_pct(th.max_vacancy)}.")
    if m.default_rate is not None and m.default_rate > th.max_default:
        vetoes.append(f"Inadimplência de {_pct(m.default_rate)} acima do limite de {_pct(th.max_default)}.")
    if m.leverage is not None and m.leverage > th.max_leverage:
        vetoes.append(f"Alavancagem de {_pct(m.leverage)} acima do limite de {_pct(th.max_leverage)}.")
    aw, sw = weights_after(ctx, m.ticker, m.segment, m.price)
    current = safe_div(ctx.values.get(m.ticker, ZERO), ctx.total) if ctx.total else ZERO
    if aw > ctx.limit_asset:
        if current > ctx.limit_asset:
            vetoes.append(f"Já representa {_pct(current)} da carteira, acima do limite de {_pct(ctx.limit_asset)} por fundo.")
        else:
            vetoes.append(f"Uma única cota levaria o fundo a {_pct(aw)} da carteira (limite {_pct(ctx.limit_asset)}).")
    if m.segment and sw > ctx.limit_segment:
        vetoes.append(f"O segmento {m.segment} passaria a {_pct(sw)} da carteira (limite {_pct(ctx.limit_segment)}).")

    if m.dy_12m is not None and m.dy_12m >= th.abnormal_yield:
        risks.append(f"Yield de {_pct(m.dy_12m)} é anormalmente alto: confira se há eventos não recorrentes ou risco de crédito.")
    if m.is_paper:
        risks.append("Fundo de recebíveis: renda sensível a juros/inflação e ao risco de crédito dos CRIs.")
    if m.dividend_cv is not None and m.dividend_cv > D("0.25"):
        risks.append(f"Rendimentos instáveis (CV {_pct(m.dividend_cv)}).")
    if m.dividend_trend is not None and th.dividend_cut < m.dividend_trend < D("-0.05"):
        risks.append(f"Rendimentos em queda moderada ({_pct(m.dividend_trend)}).")
    if m.pvp is not None and D("1.0") < m.pvp <= th.max_pvp:
        risks.append(f"Negociado com prêmio sobre o VP (P/VP {m.pvp:.2f}).".replace(".", ",", 1))
    if m.avg_daily_liquidity is not None and th.min_liquidity <= m.avg_daily_liquidity < D("1000000"):
        risks.append("Liquidez moderada: ordens grandes podem mover o preço.")
    if m.vacancy is None and m.default_rate is None and m.leverage is None:
        risks.append("Fundamentos (vacância, inadimplência, alavancagem) não avaliados: consulte o relatório gerencial.")
    return vetoes, risks


def evaluate(m: FundMetrics, ctx: PortfolioContext, profile: InvestorProfile, ref_purchase=D("1000")) -> dict:
    weights = PROFILE_WEIGHTS[profile]
    th = PROFILE_THRESHOLDS[profile]
    factors = score_factors(m, ctx, ref_purchase)
    avail = [f for f in factors if f.score is not None]
    covered = sum((weights[f.factor] for f in avail), ZERO)
    score = None
    if covered > 0:
        score = sum((f.score * weights[f.factor] for f in avail), ZERO) / covered
    vetoes, risks = vetoes_and_risks(m, ctx, th)
    if covered < th.min_coverage:
        vetoes.append(f"Dados insuficientes: só {_pct(covered)} do peso do perfil pôde ser avaliado (mínimo {_pct(th.min_coverage)}).")
    return {
        "ticker": m.ticker,
        "segment": m.segment,
        "price": m.price,
        "score": score.quantize(D("0.1")) if score is not None else None,
        "coverage": covered,
        "eligible": not vetoes,
        "vetoes": vetoes,
        "risks": risks,
        "factors": [
            {"factor": f.factor, "label": FACTOR_LABEL[f.factor], "weight": weights[f.factor],
             "score": f.score.quantize(D("0.1")) if f.score is not None else None, "detail": f.detail,
             "source": m.sources.get(f.factor)}
            for f in sorted(factors, key=lambda f: -weights[f.factor])
        ],
    }


def _justification(ev: dict) -> str:
    rated = [f for f in ev["factors"] if f["score"] is not None]
    best = sorted(rated, key=lambda f: -(f["score"] * f["weight"]))[:3]
    return "Pontos fortes para este perfil: " + "; ".join(f"{FACTOR_SHORT[f['factor']]} ({f['detail']})" for f in best) + "."


def recommend(
    candidates: list[FundMetrics], ctx: PortfolioContext, profile: InvestorProfile, budget, max_picks: int = 3
) -> dict:
    """Ranqueia, aplica vetos e distribui o orçamento respeitando os limites de concentração."""
    budget = D(budget)
    evaluations = [evaluate(m, ctx, profile, ref_purchase=min(budget, D("1000")) or D("1000")) for m in candidates]
    by_ticker = {m.ticker: m for m in candidates}
    ranked = sorted([e for e in evaluations if e["eligible"]], key=lambda e: -e["score"])

    values = dict(ctx.values)
    remaining = budget
    picks = []
    for ev in ranked:
        if len(picks) >= max_picks or remaining <= 0:
            break
        m = by_ticker[ev["ticker"]]
        price = D(m.price)
        if price > remaining:
            continue
        sim_ctx = PortfolioContext(values, ctx.segments, ctx.limit_asset, ctx.limit_segment)
        qty = int(remaining // price)
        # reduz até respeitar os limites de ativo e segmento após a compra
        while qty > 0:
            aw, sw = weights_after(sim_ctx, m.ticker, m.segment, price * qty)
            if aw <= ctx.limit_asset and (not m.segment or sw <= ctx.limit_segment):
                break
            qty -= 1
        if qty <= 0:
            continue
        cost = money(price * qty)
        remaining = money(remaining - cost)
        values[m.ticker] = values.get(m.ticker, ZERO) + cost
        total_after = sum(values.values(), ZERO)
        picks.append({
            **{k: ev[k] for k in ("ticker", "segment", "price", "score", "coverage", "risks", "factors")},
            "quantity": qty,
            "cost": cost,
            "remaining_after": remaining,
            "incremental_monthly_income": money(m.mean_monthly_per_share * qty) if m.mean_monthly_per_share else None,
            "weight_after": safe_div(values[m.ticker], total_after),
            "justification": _justification(ev),
        })

    total_after = sum(values.values(), ZERO)
    distribution = sorted(
        [{"ticker": t, "weight_before": safe_div(ctx.values.get(t, ZERO), ctx.total) if ctx.total else ZERO,
          "weight_after": safe_div(v, total_after) if total_after else ZERO} for t, v in values.items() if v > 0],
        key=lambda r: -r["weight_after"],
    )
    return {
        "profile": profile,
        "budget": money(budget),
        "allocated": money(budget - remaining),
        "remaining": money(remaining),
        "picks": picks,
        "ranking": sorted(evaluations, key=lambda e: (not e["eligible"], -(e["score"] or 0))),
        "distribution": distribution,
        "weights": {k: v for k, v in PROFILE_WEIGHTS[profile].items()},
        "thresholds": {k: getattr(PROFILE_THRESHOLDS[profile], k) for k in Thresholds.__dataclass_fields__},
    }
