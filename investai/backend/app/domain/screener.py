"""Triagem de fundos novos: filtros objetivos sobre o universo listado, antes do motor de recomendação.

A triagem só reduz o universo (centenas de fundos) a uma lista curta. A decisão continua com o motor
de recomendação, que avalia os finalistas com os mesmos fatores e vetos usados na carteira.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal

from app.domain.money import D


SEGMENT_MAP = {
    "titulos e val. mob.": "Recebíveis (CRI)", "titulos e valores mobiliarios": "Recebíveis (CRI)",
    "lajes corporativas": "Lajes corporativas", "shoppings": "Shoppings", "logistica": "Logística", "hibrido": "Híbrido",
    "residencial": "Residencial", "hospital": "Hospital", "hotel": "Hotel", "agencias de bancos": "Agências bancárias",
    "educacional": "Educacional", "outros": "Outros",
}


def normalize_segment(raw: str | None) -> str | None:
    """Segmento da CVM → rótulo usado na carteira (mantém a soma por segmento consistente)."""
    if not raw:
        return None
    return SEGMENT_MAP.get(_norm(raw).strip(), raw.strip())


@dataclass
class UniverseFund:
    ticker: str
    sub_type: str
    close: Decimal | None
    volume: Decimal | None
    name: str | None = None
    segment: str | None = None
    mandate: str | None = None
    net_worth: Decimal | None = None
    shareholders: int | None = None
    vp_per_share: Decimal | None = None


@dataclass
class ScreenCriteria:
    sub_types: set[str] = field(default_factory=lambda: {"fii"})
    segment: str | None = None  # trecho do segmento CVM, sem acento/caixa (ex.: "logist")
    min_liquidity: Decimal = D("300000")  # R$ no pregão listado (pré-filtro; média real vem do histórico)
    min_net_worth: Decimal = D("300000000")
    min_shareholders: int = 10000
    min_pvp: Decimal = D("0.70")
    max_pvp: Decimal = D("1.10")
    require_cvm: bool = False  # True = descarta fundo sem dados da CVM
    exclude: set[str] = field(default_factory=set)
    max_results: int = 15


def _norm(s: str | None) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def check(f: UniverseFund, c: ScreenCriteria) -> tuple[list[str], list[str]]:
    """(motivos de reprovação, observações)."""
    fails, notes = [], []
    if f.ticker in c.exclude:
        fails.append("já está na carteira")
    if f.sub_type not in c.sub_types:
        fails.append(f"tipo {f.sub_type} fora do filtro")
    if not f.close or f.close <= 0:
        fails.append("sem cotação")
    liq = (f.close or 0) * (f.volume or 0)
    if liq < c.min_liquidity:
        fails.append(f"liquidez R$ {D(liq) / 1000:,.0f} mil abaixo de R$ {c.min_liquidity / 1000:,.0f} mil".replace(",", "."))
    if c.segment:
        if f.segment is None:
            fails.append("segmento desconhecido (sem dados da CVM)")
        elif _norm(c.segment) not in _norm(f.segment) and _norm(c.segment) not in _norm(f.mandate):
            fails.append(f"segmento {f.segment} fora do filtro")
    has_cvm = any(v is not None for v in (f.net_worth, f.shareholders, f.vp_per_share))
    if not has_cvm:
        (fails if c.require_cvm else notes).append("sem dados da CVM: patrimônio, cotistas e P/VP não verificados")
    if f.net_worth is not None and f.net_worth < c.min_net_worth:
        fails.append(f"patrimônio de R$ {f.net_worth / 1_000_000:,.0f} mi abaixo do mínimo".replace(",", "."))
    if f.shareholders is not None and f.shareholders < c.min_shareholders:
        fails.append(f"{f.shareholders:,} cotistas, abaixo de {c.min_shareholders:,}".replace(",", "."))
    if f.vp_per_share and f.close:
        pvp = f.close / f.vp_per_share
        if pvp < c.min_pvp:
            fails.append(f"P/VP {pvp:.2f} muito baixo: pode sinalizar problema no fundo".replace(".", ","))
        elif pvp > c.max_pvp:
            fails.append(f"P/VP {pvp:.2f} acima de {c.max_pvp:.2f}".replace(".", ","))
    return fails, notes


def _category(reason: str) -> str:
    for prefix, label in (("já está", "já na carteira"), ("tipo", "tipo de fundo"), ("sem cotação", "sem cotação"),
                          ("liquidez", "liquidez baixa"), ("segmento", "segmento"), ("sem dados da CVM", "sem dados da CVM"),
                          ("patrimônio", "patrimônio baixo"), ("P/VP", "P/VP fora da faixa")):
        if reason.startswith(prefix):
            return label
    return "poucos cotistas" if "cotistas" in reason else reason


def screen(universe: list[UniverseFund], c: ScreenCriteria) -> dict:
    passed: list[dict] = []
    reasons: dict[str, int] = {}
    for f in universe:
        fails, notes = check(f, c)
        if fails:
            k = _category(fails[0])
            reasons[k] = reasons.get(k, 0) + 1
            continue
        passed.append({
            "ticker": f.ticker, "name": f.name, "sub_type": f.sub_type, "segment": f.segment, "close": f.close,
            "listed_liquidity": (f.close or 0) * (f.volume or 0), "net_worth": f.net_worth, "shareholders": f.shareholders,
            "pvp": (f.close / f.vp_per_share) if (f.close and f.vp_per_share) else None, "notes": notes,
        })
    passed.sort(key=lambda r: -r["listed_liquidity"])
    return {
        "universe_size": len(universe),
        "passed": len(passed),
        "shortlist": passed[: c.max_results],
        "rejected_by_first_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
