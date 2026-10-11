"""Importação de CSV/Excel com detecção de formato, normalização e chave de deduplicação.

Formatos suportados:
- investai: modelo próprio (ver GET /api/imports/template).
- b3_negociacao: "Extrato de Negociação" da Área do Investidor B3 (compras e vendas).
- b3_movimentacao: "Extrato de Movimentação" da B3 — importa Rendimento e Amortização. Liquidações
  de compra/venda desse extrato são ignoradas de propósito (duplicariam o extrato de negociação).
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from app.domain.enums import ACQUISITION_TYPES, SPLIT_TYPES, TxnStatus, TxnType
from app.domain.money import D

TEMPLATE_COLUMNS = [
    "data", "tipo", "status", "ticker", "quantidade", "preco_unitario", "taxas", "valor_total",
    "proporcao_de", "proporcao_para", "observacao",
]

TYPE_ALIASES = {
    "COMPRA": TxnType.BUY, "VENDA": TxnType.SELL, "SUBSCRICAO": TxnType.SUBSCRIPTION,
    "DIREITO_PREFERENCIA": TxnType.PREEMPTIVE_RIGHT, "DIREITO DE PREFERENCIA": TxnType.PREEMPTIVE_RIGHT,
    "SOBRAS": TxnType.SUBSCRIPTION_LEFTOVER, "AMORTIZACAO": TxnType.AMORTIZATION,
    "DESDOBRAMENTO": TxnType.SPLIT, "DESDOBRO": TxnType.SPLIT, "GRUPAMENTO": TxnType.REVERSE_SPLIT,
    "DIVIDENDO": TxnType.DIVIDEND, "RENDIMENTO": TxnType.DIVIDEND, "TAXA": TxnType.FEE,
    "APORTE": TxnType.CASH_DEPOSIT, "RETIRADA": TxnType.CASH_WITHDRAWAL, "SALDO_INICIAL": TxnType.OPENING_BALANCE,
}
STATUS_ALIASES = {"CONFIRMADA": TxnStatus.CONFIRMED, "PENDENTE": TxnStatus.PENDING, "PLANEJADA": TxnStatus.PLANNED}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().upper()


def parse_br_number(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int,)):
        return D(v)
    if isinstance(v, float):
        return D(repr(v))
    s = str(v).strip().replace("R$", "").replace(" ", "")
    if s in ("", "-", "--"):
        return None
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return D(s)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"número inválido: {v!r}") from e


def parse_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, pd.Timestamp):
        return v.date()
    s = str(v).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"data inválida: {v!r}")


def clean_ticker(raw: str) -> str:
    t = _norm(raw).split(" - ")[0].split()[0]
    # mercado fracionário: KNRI11F -> KNRI11
    if re.fullmatch(r"[A-Z]{4}\d{1,2}F", t):
        t = t[:-1]
    return t


@dataclass
class ParsedRow:
    row_number: int
    data: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    hash: str | None = None


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content), dtype=object)
    text = content.decode("utf-8-sig", errors="replace")
    sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def detect_format(df: pd.DataFrame) -> str:
    cols = {_norm(c) for c in df.columns}
    if {"DATA DO NEGOCIO", "TIPO DE MOVIMENTACAO", "CODIGO DE NEGOCIACAO"} <= cols:
        return "b3_negociacao"
    if {"ENTRADA/SAIDA", "MOVIMENTACAO", "PRODUTO"} <= cols:
        return "b3_movimentacao"
    if {"DATA", "TIPO", "TICKER"} <= cols:
        return "investai"
    raise ValueError("Formato não reconhecido. Use o modelo InvestAI ou extratos de Negociação/Movimentação da B3.")


def _get(row: dict, *names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            v = row[n]
            if isinstance(v, float) and pd.isna(v):
                continue
            return v
    return None


def _parse_investai(r: dict) -> dict:
    tipo = _norm(_get(r, "TIPO") or "")
    ttype = TYPE_ALIASES.get(tipo) or TxnType(tipo)
    st = _norm(_get(r, "STATUS") or "CONFIRMADA")
    status = STATUS_ALIASES.get(st) or TxnStatus(st)
    ticker = _get(r, "TICKER")
    return {
        "trade_date": parse_date(_get(r, "DATA")),
        "type": ttype,
        "status": status,
        "ticker": clean_ticker(ticker) if ticker else None,
        "quantity": parse_br_number(_get(r, "QUANTIDADE")),
        "unit_price": parse_br_number(_get(r, "PRECO_UNITARIO", "PRECO UNITARIO")),
        "fees": parse_br_number(_get(r, "TAXAS")) or D(0),
        "total_amount": parse_br_number(_get(r, "VALOR_TOTAL", "VALOR TOTAL")),
        "ratio_from": parse_br_number(_get(r, "PROPORCAO_DE")),
        "ratio_to": parse_br_number(_get(r, "PROPORCAO_PARA")),
        "notes": _get(r, "OBSERVACAO"),
    }


def _parse_b3_negociacao(r: dict) -> dict:
    side = _norm(_get(r, "TIPO DE MOVIMENTACAO") or "")
    if side not in ("COMPRA", "VENDA"):
        raise ValueError(f"movimentação desconhecida: {side}")
    qty = parse_br_number(_get(r, "QUANTIDADE"))
    price = parse_br_number(_get(r, "PRECO"))
    return {
        "trade_date": parse_date(_get(r, "DATA DO NEGOCIO")),
        "type": TxnType.BUY if side == "COMPRA" else TxnType.SELL,
        "status": TxnStatus.CONFIRMED,
        "ticker": clean_ticker(_get(r, "CODIGO DE NEGOCIACAO")),
        "quantity": qty,
        "unit_price": price,
        "fees": D(0),
        "total_amount": None,
        "notes": f"Importado B3 Negociação ({_get(r, 'INSTITUICAO') or ''})".strip(),
    }


B3_MOV_MAP = {"RENDIMENTO": TxnType.DIVIDEND, "AMORTIZACAO": TxnType.AMORTIZATION}


def _parse_b3_movimentacao(r: dict) -> dict | str:
    mov = _norm(_get(r, "MOVIMENTACAO") or "")
    ttype = B3_MOV_MAP.get(mov)
    if ttype is None:
        return f"movimentação '{mov.title()}' não importada automaticamente (registre manualmente se necessário)"
    if _norm(_get(r, "ENTRADA/SAIDA") or "") != "CREDITO":
        return "lançamento a débito ignorado"
    total = parse_br_number(_get(r, "VALOR DA OPERACAO"))
    if total is None:
        return "provento sem valor (provisionado) — aguardando pagamento"
    return {
        "trade_date": parse_date(_get(r, "DATA")),
        "type": ttype,
        "status": TxnStatus.CONFIRMED,
        "ticker": clean_ticker(_get(r, "PRODUTO")),
        "quantity": parse_br_number(_get(r, "QUANTIDADE")),
        "unit_price": parse_br_number(_get(r, "PRECO UNITARIO")),
        "fees": D(0),
        "total_amount": total,
        "notes": "Importado B3 Movimentação",
    }


PARSERS = {"investai": _parse_investai, "b3_negociacao": _parse_b3_negociacao, "b3_movimentacao": _parse_b3_movimentacao}


def _validate(d: dict) -> list[str]:
    errs = []
    t = d["type"]
    if t in ACQUISITION_TYPES | SPLIT_TYPES | {TxnType.SELL, TxnType.DIVIDEND, TxnType.AMORTIZATION} and not d.get("ticker"):
        errs.append("ticker obrigatório")
    if t in ACQUISITION_TYPES | {TxnType.SELL}:
        if not d.get("quantity") or d["quantity"] <= 0:
            errs.append("quantidade deve ser positiva")
        if t == TxnType.SELL and d.get("unit_price") is None and d.get("total_amount") is None:
            errs.append("venda exige preço ou valor total")
    if t in SPLIT_TYPES and (not d.get("ratio_from") or not d.get("ratio_to")):
        errs.append("desdobramento/grupamento exige proporcao_de e proporcao_para")
    if t in {TxnType.DIVIDEND, TxnType.AMORTIZATION, TxnType.FEE, TxnType.CASH_DEPOSIT, TxnType.CASH_WITHDRAWAL}:
        if d.get("total_amount") is None or d["total_amount"] < 0:
            errs.append("valor_total obrigatório e não negativo")
    return errs


def fingerprint(d: dict) -> str:
    parts = [
        d["trade_date"].isoformat(), str(d["type"]), d.get("ticker") or "",
        _num(d.get("quantity")), _num(d.get("unit_price")), _num(d.get("total_amount")),
    ]
    return "|".join(parts)


def _num(v) -> str:
    return "" if v is None else format(D(v).normalize(), "f")


def parse_file(content: bytes, filename: str, fmt: str | None = None) -> tuple[str, list[ParsedRow]]:
    df = read_table(content, filename)
    fmt = fmt or detect_format(df)
    parser = PARSERS[fmt]
    df.columns = [_norm(c) for c in df.columns]
    rows: list[ParsedRow] = []
    seen: dict[str, int] = {}
    for i, rec in enumerate(df.to_dict(orient="records"), start=2):  # linha 1 = cabeçalho
        pr = ParsedRow(row_number=i)
        try:
            out = parser(rec)
            if isinstance(out, str):
                pr.skipped_reason = out
                rows.append(pr)
                continue
            pr.data = out
            pr.errors = _validate(out)
        except Exception as e:  # noqa: BLE001
            pr.errors = [str(e)]
        if pr.data and not pr.errors:
            fp = fingerprint(pr.data)
            # Duas linhas idênticas no mesmo arquivo são legítimas (ex.: duas compras iguais no dia):
            # o índice de ocorrência entra na chave para distinguir.
            n = seen.get(fp, 0)
            seen[fp] = n + 1
            pr.hash = hashlib.sha256(f"{fp}#{n}".encode()).hexdigest()
        rows.append(pr)
    return fmt, rows


def template_csv() -> str:
    lines = [
        ";".join(TEMPLATE_COLUMNS),
        "15/01/2026;COMPRA;CONFIRMADA;KNRI11;2;158,40;0,00;;;;Exemplo — apague esta linha",
        "14/02/2026;RENDIMENTO;CONFIRMADA;KNRI11;10;;;10,00;;;Exemplo",
        "01/03/2026;DESDOBRAMENTO;CONFIRMADA;XXXX11;;;;;1;10;Exemplo 1:10",
    ]
    return "\n".join(lines) + "\n"
