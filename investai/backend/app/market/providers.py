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
from zoneinfo import ZoneInfo

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
    volume: Decimal | None = None


@dataclass
class DividendResult:
    ticker: str
    amount_per_share: Decimal
    com_date: date
    payment_date: date
    kind: str
    source: str
    ex_date: date | None = None


@dataclass
class FetchReport:
    quotes: list[QuoteResult] = field(default_factory=list)
    dividends: list[DividendResult] = field(default_factory=list)
    missing: dict[str, str] = field(default_factory=dict)  # ticker -> motivo


class MarketDataProvider(Protocol):
    name: str
    capabilities: set[str]

    def fetch(self, tickers: list[str], with_dividends: bool = False, history_range: str | None = None) -> FetchReport: ...


class ManualProvider:
    """Sem rede: cotações e proventos são cadastrados pelo usuário."""

    name = "MANUAL"
    capabilities: set[str] = set()

    def fetch(self, tickers, with_dividends=False, history_range=None) -> FetchReport:
        return FetchReport(missing={t: "Provedor manual: cadastre a cotação na tela do ativo." for t in tickers})


BR_TZ = ZoneInfo("America/Sao_Paulo")


def _parse_iso_date(v) -> date | None:
    """BRAPI devolve datas em UTC (ex.: 2026-12-21T03:00:00Z = meia-noite de Brasília)."""
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(BR_TZ).date()
    except ValueError:
        return None


class BrapiProvider:
    name = "BRAPI"
    capabilities = {"quote", "dividends?"}

    def __init__(self, token: str | None, base_url: str, timeout: float = 10.0, client: httpx.Client | None = None):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)

    def fetch(self, tickers, with_dividends=False, history_range=None) -> FetchReport:
        """Formato validado contra a resposta real de /quote (out/2026): regularMarketPrice/Time/Volume,
        historicalDataPrice[{date (epoch), close, volume}] e dividendsData.cashDividends[{rate,
        lastDatePrior (data-com), exDate, paymentDate, label}]."""
        rep = FetchReport()
        if not self.token:
            rep.missing = {t: "BRAPI sem token configurado (INVESTAI_BRAPI_TOKEN)." for t in tickers}
            return rep
        for t in tickers:  # uma requisição por ativo: o plano gratuito limita múltiplos tickers por chamada
            params = {}
            if with_dividends:
                params["dividends"] = "true"
            if history_range:
                params.update({"range": history_range, "interval": "1d"})
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
            vol = item.get("regularMarketVolume")
            if price is None or qdate is None:
                rep.missing[t] = "Cotação ausente na resposta."
            else:
                rep.quotes.append(QuoteResult(t, qdate, D(price), self.name, D(vol) if vol is not None else None))
            if history_range:
                hist = item.get("historicalDataPrice")
                if not hist:
                    rep.missing.setdefault(t, f"Histórico ({history_range}) indisponível neste plano/endpoint.")
                for h in hist or []:
                    if h.get("close") is None or h.get("date") is None:
                        continue
                    d = datetime.fromtimestamp(int(h["date"]), BR_TZ).date()
                    if qdate and d == qdate:
                        continue  # o pregão corrente já veio em regularMarket*
                    v = h.get("volume")
                    rep.quotes.append(QuoteResult(t, d, D(h["close"]), self.name, D(v) if v is not None else None))
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
                    rep.dividends.append(DividendResult(t, D(rate), com, pay, kind, self.name, _parse_iso_date(c.get("exDate"))))
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
    if settings.market_provider == "brapi" or (settings.market_provider == "auto" and settings.brapi_token):
        return BrapiProvider(settings.brapi_token, settings.brapi_base_url, settings.http_timeout_seconds)
    return ManualProvider()
