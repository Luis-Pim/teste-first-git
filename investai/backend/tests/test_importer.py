from datetime import date
from decimal import Decimal as Dec

import pandas as pd
import io

from app.domain.enums import TxnType as T
from app.domain.importer import clean_ticker, parse_br_number, parse_file, template_csv


def test_br_number_parsing():
    assert parse_br_number("R$ 1.046,80") == Dec("1046.80")
    assert parse_br_number("9,7") == Dec("9.7")
    assert parse_br_number("-") is None
    assert parse_br_number(10.5) == Dec("10.5")


def test_fractional_ticker_normalized():
    assert clean_ticker("KNRI11F") == "KNRI11"
    assert clean_ticker("KNRI11 - KINEA RENDA IMOBILIARIA FDO INV IMOB - FII") == "KNRI11"


def test_template_parses_and_exact_duplicates_in_same_file_get_distinct_hashes():
    csv = template_csv() + "15/01/2026;COMPRA;CONFIRMADA;KNRI11;2;158,40;0,00;;;;Exemplo — apague esta linha\n"
    fmt, rows = parse_file(csv.encode(), "x.csv")
    assert fmt == "investai"
    assert not any(r.errors for r in rows)
    assert rows[0].data["type"] == T.BUY and rows[2].data["type"] == T.SPLIT
    assert rows[0].hash != rows[3].hash
    # reimportar o mesmo arquivo gera os mesmos hashes (deduplicação estável)
    _, again = parse_file(csv.encode(), "x.csv")
    assert [r.hash for r in rows] == [r.hash for r in again]


def test_b3_negociacao_csv():
    csv = (
        "Data do Negócio;Tipo de Movimentação;Mercado;Prazo/Vencimento;Instituição;Código de Negociação;Quantidade;Preço;Valor\n"
        "02/09/2026;Compra;Mercado Fracionário;-;XP;MXRF11F;10;9,70;97,00\n"
        "03/09/2026;Venda;Mercado à Vista;-;XP;KNRI11;1;150,00;150,00\n"
    )
    fmt, rows = parse_file(csv.encode("utf-8"), "neg.csv")
    assert fmt == "b3_negociacao"
    assert rows[0].data["ticker"] == "MXRF11" and rows[0].data["trade_date"] == date(2026, 9, 2)
    assert rows[1].data["type"] == T.SELL


def test_b3_movimentacao_xlsx_imports_income_and_skips_settlements():
    df = pd.DataFrame(
        [
            ["Credito", "14/09/2026", "Rendimento", "KNRI11 - KINEA RENDA", "XP", 8, 1.0, 8.0],
            ["Credito", "02/09/2026", "Transferência - Liquidação", "MXRF11 - MAXI RENDA", "XP", 10, 9.7, 97.0],
            ["Credito", "20/10/2026", "Rendimento", "XPML11 - XP MALLS", "XP", 17, "-", "-"],
        ],
        columns=["Entrada/Saída", "Data", "Movimentação", "Produto", "Instituição", "Quantidade", "Preço unitário", "Valor da Operação"],
    )
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    fmt, rows = parse_file(buf.getvalue(), "mov.xlsx")
    assert fmt == "b3_movimentacao"
    assert rows[0].data["type"] == T.DIVIDEND and rows[0].data["total_amount"] == Dec("8.0")
    assert rows[1].skipped_reason and rows[2].skipped_reason


def test_invalid_rows_reported():
    csv = "data;tipo;ticker;quantidade;preco_unitario\n31/02/2026;COMPRA;KNRI11;1;10\n01/02/2026;COMPRA;;1;10\n"
    _, rows = parse_file(csv.encode(), "bad.csv")
    assert rows[0].errors and rows[1].errors == ["ticker obrigatório"]
