"""Informe mensal de FII (CVM Dados Abertos) → valor patrimonial por cota.

Arquivo: https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/inf_mensal_fii_{ano}.zip
contendo, entre outros, `inf_mensal_fii_geral_{ano}.csv` (CNPJ, data de referência, ISIN) e
`inf_mensal_fii_complemento_{ano}.csv` (patrimônio líquido, cotas emitidas, valor patrimonial da cota).

Os nomes de coluna mudaram com a Resolução CVM 175 (ex.: CNPJ_Fundo → CNPJ_Fundo_Classe), por isso a
leitura localiza as colunas por trecho do nome. O vínculo ticker ↔ fundo usa o ISIN da cota de FII,
que segue o padrão BR + raiz do ticker + "CTF" (ex.: KNRI11 → BRKNRICTF…).

PENDENTE: validar contra o arquivo real (o domínio estava bloqueado na sessão de desenvolvimento).
"""

from __future__ import annotations

import io
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import httpx
import pandas as pd

from app.domain.money import D
from app.market.providers import ProviderError

CVM_URL = "https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/inf_mensal_fii_{year}.zip"


@dataclass
class CvmIndicator:
    ticker: str
    ref_date: date
    vp_per_share: Decimal
    cnpj: str
    source_url: str


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()


def _col(df: pd.DataFrame, *needles: str, exclude: tuple[str, ...] = ()) -> str | None:
    for c in df.columns:
        n = _norm(c)
        if all(x in n for x in needles) and not any(x in n for x in exclude):
            return c
    return None


def _num(v) -> Decimal | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return D(s)
    except InvalidOperation:
        return None


def ticker_root(ticker: str) -> str:
    return ticker[:4].upper()


def parse_zip(content: bytes, tickers: list[str], source_url: str = "") -> tuple[list[CvmIndicator], list[str]]:
    """Devolve (indicadores encontrados, avisos)."""
    warnings: list[str] = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = zf.namelist()
        geral_name = next((n for n in names if "geral" in n.lower()), None)
        comp_name = next((n for n in names if "complemento" in n.lower()), None)
        if not geral_name or not comp_name:
            raise ProviderError(f"Arquivos esperados não encontrados no ZIP da CVM: {names}")
        read = lambda n: pd.read_csv(zf.open(n), sep=";", encoding="latin-1", dtype=str, keep_default_na=False)  # noqa: E731
        geral, comp = read(geral_name), read(comp_name)

    g_cnpj, g_isin = _col(geral, "CNPJ"), _col(geral, "ISIN")
    c_cnpj, c_date = _col(comp, "CNPJ"), _col(comp, "DATA", "REFERENCIA")
    c_vp = _col(comp, "VALOR", "PATRIMONIAL", "COTA")
    c_pl, c_cotas = _col(comp, "PATRIMONIO", "LIQUIDO"), _col(comp, "COTAS", "EMITIDAS")
    c_ver = _col(comp, "VERSAO")
    if not all([g_cnpj, g_isin, c_cnpj, c_date]) or not (c_vp or (c_pl and c_cotas)):
        raise ProviderError("Layout do informe mensal da CVM não reconhecido (colunas de CNPJ, ISIN, data ou VP ausentes).")

    roots = {ticker_root(t): t for t in tickers}
    cnpj_to_ticker: dict[str, str] = {}
    for _, r in geral.iterrows():
        isin = str(r[g_isin]).strip().upper()
        if len(isin) >= 9 and isin.startswith("BR") and isin[6:9] == "CTF" and isin[2:6] in roots:
            cnpj_to_ticker[str(r[g_cnpj]).strip()] = roots[isin[2:6]]
    missing = sorted(set(tickers) - set(cnpj_to_ticker.values()))
    if missing:
        warnings.append(f"Sem correspondência por ISIN no informe da CVM: {', '.join(missing)}.")

    best: dict[tuple[str, date], tuple[int, Decimal, str]] = {}
    for _, r in comp.iterrows():
        cnpj = str(r[c_cnpj]).strip()
        t = cnpj_to_ticker.get(cnpj)
        if not t:
            continue
        try:
            ref = datetime.strptime(str(r[c_date]).strip()[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        vp = _num(r[c_vp]) if c_vp else None
        if (vp is None or vp <= 0) and c_pl and c_cotas:
            pl, cotas = _num(r[c_pl]), _num(r[c_cotas])
            vp = pl / cotas if pl and cotas else None
        if vp is None or vp <= 0:
            continue
        ver = int(_num(r[c_ver]) or 0) if c_ver else 0
        key = (t, ref)
        if key not in best or ver >= best[key][0]:  # reapresentações: fica a versão mais recente
            best[key] = (ver, vp, cnpj)
    out = [CvmIndicator(t, ref, vp, cnpj, source_url) for (t, ref), (_, vp, cnpj) in sorted(best.items())]
    return out, warnings


def fetch(year: int, tickers: list[str], timeout: float = 60.0, client: httpx.Client | None = None) -> tuple[list[CvmIndicator], list[str]]:
    url = CVM_URL.format(year=year)
    client = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        r = client.get(url)
    except httpx.HTTPError as e:
        raise ProviderError(f"CVM indisponível: {e.__class__.__name__}") from e
    if r.status_code != 200:
        raise ProviderError(f"CVM HTTP {r.status_code} em {url}")
    return parse_zip(r.content, tickers, url)
