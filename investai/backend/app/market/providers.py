"""Adaptadores de dados de mercado.

Cada provedor declara explicitamente o que oferece (`capabilities`). Dados ausentes são reportados como
ausentes — nunca preenchidos com estimativas.

Estado verificado (out/2026), a confirmar na ativação da Fase 2:
- BRAPI (brapi.dev): cotação via /quote/{tickers} exige token para a maioria dos ativos; o plano
  gratuito tem limite mensal de requisições e restringe módulos. Proventos (`dividends=true`) e
  histórico longo podem exigir plano pago — por isso o adaptador trata a ausência do bloco
  `dividendsData` como "indisponível", e não como "sem proventos".
- BCB/SGS: público, sem token. CDI diário = série 12; Selic = 11; IPCA mensal = 433.
- IBGE/SIDRA: IPCA oficial (tabela 1737) — alternativa ao SGS 433.
- B3 / CVM: informes mensais de FIIs (dados.cvm.gov.br, "inf_mensal_fii") trazem patrimônio,
  cotas emitidas, VP/cota e indicadores; vacância e inadimplência vêm dos relatórios gerenciais
  (PDF) das administradoras → candidatos a RAG na Fase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol

import httpx

from app.domain.money import D


class ProviderError(RuntimeError):
    pass


@dataclass
class QuoteResult:
    ticker: str
    quote_date: date
    close: Decimal
    source: str


@dataclass
class DividendResult:
    ticker: str
    amount_per_share: Decimal
    com_date: date
    payment_date: date
    kind: str
    source: str


@dataclass
class FetchReport:
    quotes: list[QuoteResult] = field(default_factory=list)
    dividends: list[DividendResult] = field(default_factory=list)
    missing: dict[str, str] = field(default_factory=dict)  # ticker -> motivo


class MarketDataProvider(Protocol):
    name: str
    capabilities: set[str]

    def fetch(self, tickers: list[str], with_dividends: bool = False) -> FetchReport: ...


class ManualProvider:
    """Sem rede: cotações e proventos são cadastrados pelo usuário."""

    name = "MANUAL"
    capabilities: set[str] = set()

    def fetch(self, tickers, with_dividends=False) -> FetchReport:
        return FetchReport(missing={t: "Provedor manual: cadastre a cotação na tela do ativo." for t in tickers})


def _parse_iso_date(v) -> date | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).date()
    except ValueError:
        return None


class BrapiProvider:
    name = "BRAPI"
    capabilities = {"quote", "dividends?"}

    def __init__(self, token: str | None, base_url: str, timeout: float = 10.0, client: httpx.Client | None = None):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)

    def fetch(self, tickers, with_dividends=False) -> FetchReport:
        rep = FetchReport()
        if not self.token:
            rep.missing = {t: "BRAPI sem token configurado (INVESTAI_BRAPI_TOKEN)." for t in tickers}
            return rep
        for t in tickers:  # uma requisição por ativo: o plano gratuito limita múltiplos tickers por chamada
            params = {"dividends": "true"} if with_dividends else {}
            try:
                r = self.client.get(f"{self.base_url}/quote/{t}", params=params, headers={"Authorization": f"Bearer {self.token}"})
            except httpx.HTTPError as e:
                rep.missing[t] = f"Falha de rede: {e.__class__.__name__}"
                continue
            if r.status_code != 200:
                rep.missing[t] = f"HTTP {r.status_code}: {r.text[:160]}"
                continue
            results = (r.json() or {}).get("results") or []
            if not results:
                rep.missing[t] = "Ativo não retornado pela BRAPI."
                continue
            item = results[0]
            price = item.get("regularMarketPrice")
            qdate = _parse_iso_date(item.get("regularMarketTime"))
            if price is None or qdate is None:
                rep.missing[t] = "Cotação ausente na resposta."
            else:
                rep.quotes.append(QuoteResult(t, qdate, D(price), self.name))
            if with_dividends:
                dd = item.get("dividendsData")
                if not dd:
                    rep.missing.setdefault(t, "Proventos indisponíveis neste plano/endpoint da BRAPI.")
                    continue
                for c in dd.get("cashDividends") or []:
                    com = _parse_iso_date(c.get("lastDatePrior"))
                    pay = _parse_iso_date(c.get("paymentDate"))
                    rate = c.get("rate")
                    if not (com and pay and rate):
                        continue
                    label = (c.get("label") or "").upper()
                    kind = "AMORTIZACAO" if "AMORT" in label else "RENDIMENTO"
                    rep.dividends.append(DividendResult(t, D(rate), com, pay, kind, self.name))
        return rep


class BcbSgsClient:
    """Séries do Banco Central (SGS). Público e gratuito."""

    SERIES = {"CDI": 12, "SELIC": 11, "IPCA": 433}

    def __init__(self, base_url: str, timeout: float = 10.0, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)

    def fetch(self, code: str, start: date, end: date) -> list[tuple[date, Decimal]]:
        sid = self.SERIES[code]
        url = f"{self.base_url}/bcdata.sgs.{sid}/dados"
        params = {"formato": "json", "dataInicial": start.strftime("%d/%m/%Y"), "dataFinal": end.strftime("%d/%m/%Y")}
        try:
            r = self.client.get(url, params=params)
        except httpx.HTTPError as e:
            raise ProviderError(f"BCB indisponível: {e.__class__.__name__}") from e
        if r.status_code != 200:
            raise ProviderError(f"BCB HTTP {r.status_code}")
        return [(datetime.strptime(p["data"], "%d/%m/%Y").date(), D(p["valor"].replace(",", "."))) for p in r.json()]


def get_provider(settings) -> MarketDataProvider:
    if settings.market_provider == "brapi":
        return BrapiProvider(settings.brapi_token, settings.brapi_base_url, settings.http_timeout_seconds)
    return ManualProvider()
