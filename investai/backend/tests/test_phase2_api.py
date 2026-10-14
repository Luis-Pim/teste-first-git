import io
import zipfile
from datetime import date
from decimal import Decimal as Dec

import httpx

from app.core.db import SessionLocal
from app.market import cvm
from app.market.providers import BrapiProvider
from app.seed import seed_demo_market, seed_portfolio
from app.services import market_service

Q = {"as_of": "2026-10-11"}


def _seed():
    with SessionLocal() as db:
        seed_portfolio(db)
        seed_demo_market(db, date(2026, 10, 11))


def test_recommendations_with_demo_data(client):
    _seed()
    r = client.get("/api/recommendations", params={**Q, "profile": "EQUILIBRADO"}).json()
    assert r["budget"] == "300.00" and r["budget_source"].startswith("orçamento disponível")
    assert any("DEMO" in n for n in r["notes"])
    by = {e["ticker"]: e for e in r["ranking"]}
    # KNCR11 (28,8%) e XPML11 (29,2%) já estão acima do limite de 25%
    assert not by["KNCR11"]["eligible"] and not by["XPML11"]["eligible"]
    assert all(p["ticker"] not in ("KNCR11", "XPML11") for p in r["picks"])
    assert r["picks"], r["ranking"]
    pick = r["picks"][0]
    assert Dec(pick["cost"]) <= Dec("300") and pick["justification"] and pick["factors"]
    assert Dec(r["allocated"]) + Dec(r["remaining"]) == Dec("300.00")
    assert all(f["source"] for f in pick["factors"] if f["score"] is not None and f["factor"] != "diversification")


def test_recommendations_custom_budget_and_profile(client):
    _seed()
    r = client.get("/api/recommendations", params={**Q, "profile": "CONSERVADOR", "budget": "1000"}).json()
    assert r["profile"] == "CONSERVADOR" and r["budget"] == "1000.00"
    # conservador exige P/VP ≤ 1,05 e liquidez ≥ R$ 500 mil/dia
    for e in r["ranking"]:
        if e["ticker"] == "VILG11":  # 15 mil cotas × R$ 92,30 ≈ R$ 1,4 mi/dia
            assert not any("Liquidez baixa" in v for v in e["vetoes"])


def test_indicator_crud_and_validation(client):
    r = client.post("/api/indicators", json={"ticker": "KNRI11", "ref_date": "2026-08-31", "vp_per_share": "160.10", "source": "CVM"})
    assert r.status_code == 201 and Dec(r.json()["vp_per_share"]) == Dec("160.10")
    # mesma data e fonte atualiza em vez de duplicar
    client.post("/api/indicators", json={"ticker": "KNRI11", "ref_date": "2026-08-31", "vacancy": "0.07", "source": "CVM"})
    rows = client.get("/api/indicators", params={"ticker": "KNRI11"}).json()
    assert len(rows) == 1 and Dec(rows[0]["vacancy"]) == Dec("0.07") and Dec(rows[0]["vp_per_share"]) == Dec("160.10")
    assert client.post("/api/indicators", json={"ticker": "KNRI11", "ref_date": "2026-08-31"}).status_code == 422
    assert client.post("/api/indicators", json={"ticker": "KNRI11", "ref_date": "2026-08-31", "vacancy": "1.5"}).status_code == 422
    assert client.delete(f"/api/indicators/{rows[0]['id']}").status_code == 204


# --- BRAPI: resposta no formato real capturado em 14/10/2026 (PETR4, endpoint /quote com range e dividends)
BRAPI_REAL_SHAPE = {
    "results": [{
        "symbol": "KNRI11", "regularMarketPrice": 152.1, "regularMarketTime": "2026-10-14T14:03:30.000Z", "regularMarketVolume": 25000,
        "historicalDataPrice": [
            {"date": 1789354800, "open": 150.0, "high": 151.0, "low": 149.5, "close": 150.4, "volume": 31000, "adjustedClose": 150.4},
            {"date": 1789441200, "open": 150.4, "high": 152.0, "low": 150.0, "close": 151.2, "volume": 28000, "adjustedClose": 151.2},
        ],
        "dividendsData": {"cashDividends": [{
            "assetIssued": "BRKNRICTF003", "paymentDate": "2026-10-14T03:00:00.000Z", "rate": 1.0, "label": "RENDIMENTO",
            "lastDatePrior": "2026-09-30T03:00:00.000Z", "exDate": "2026-10-01T03:00:00.000Z", "approvedOn": "2026-09-30T03:00:00.000Z",
        }], "stockDividends": [], "subscriptions": []},
    }]
}


def test_brapi_history_volume_and_dates_in_brasilia_time():
    seen = {}

    def handler(req):
        seen.update(dict(req.url.params))
        return httpx.Response(200, json=BRAPI_REAL_SHAPE)

    rep = BrapiProvider("tkn", "https://brapi.test/api", client=httpx.Client(transport=httpx.MockTransport(handler))).fetch(
        ["KNRI11"], with_dividends=True, history_range="1y")
    assert seen == {"dividends": "true", "range": "1y", "interval": "1d"}
    assert len(rep.quotes) == 3 and rep.quotes[0].volume == Dec("25000")
    d = rep.dividends[0]
    # 03:00Z = meia-noite em Brasília: a data não pode "voltar" um dia
    assert (d.com_date, d.ex_date, d.payment_date) == (date(2026, 9, 30), date(2026, 10, 1), date(2026, 10, 14))
    assert not rep.missing


def test_refresh_persists_volume_and_liquidity_feeds_engine(client):
    client.post("/api/transactions", json={"type": "BUY", "trade_date": "2026-09-01", "ticker": "KNRI11", "quantity": 1, "unit_price": "100"})
    p = BrapiProvider("tkn", "https://brapi.test/api", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=BRAPI_REAL_SHAPE))))
    with SessionLocal() as db:
        res = market_service.refresh(db, p, ["KNRI11"], 15, with_dividends=True, history_range="1y")
    assert res["quotes_saved"] == 3 and res["dividends_saved"] == 1
    q = client.get("/api/quotes", params={"ticker": "KNRI11"}).json()
    assert len(q) == 3


# --- CVM: ZIP sintético no layout pós-Resolução 175

def _cvm_zip():
    geral = (
        "CNPJ_Fundo_Classe;Data_Referencia;Versao;Nome_Fundo_Classe;Codigo_ISIN\n"
        "12.005.956/0001-65;2026-08-01;1;KINEA RENDA IMOBILIARIA;BRKNRICTF003\n"
        "99.999.999/0001-99;2026-08-01;1;OUTRO FUNDO;BRXXXXCTF001\n"
    )
    comp = (
        "CNPJ_Fundo_Classe;Data_Referencia;Versao;Patrimonio_Liquido;Cotas_Emitidas;Valor_Patrimonial_Cotas\n"
        "12.005.956/0001-65;2026-07-01;1;4000000000.00;25000000;160.00\n"
        "12.005.956/0001-65;2026-08-01;1;4000000000.00;25000000;159.00\n"
        "12.005.956/0001-65;2026-08-01;2;4005000000.00;25000000;160.20\n"
        "99.999.999/0001-99;2026-08-01;1;100;1;100\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("inf_mensal_fii_geral_2026.csv", geral.encode("latin-1"))
        z.writestr("inf_mensal_fii_complemento_2026.csv", comp.encode("latin-1"))
    return buf.getvalue()


def test_cvm_parse_matches_isin_and_keeps_latest_version():
    rows, warnings = cvm.parse_zip(_cvm_zip(), ["KNRI11", "MXRF11"])
    assert [(r.ticker, r.ref_date, r.vp_per_share) for r in rows] == [
        ("KNRI11", date(2026, 7, 1), Dec("160.00")),
        ("KNRI11", date(2026, 8, 1), Dec("160.20")),  # versão 2 (reapresentação) substitui a 1
    ]
    assert warnings == ["Sem correspondência por ISIN no informe da CVM: MXRF11."]


def test_cvm_ingest_writes_indicators_and_cnpj(client):
    client.post("/api/assets", json={"ticker": "KNRI11"})
    with SessionLocal() as db:
        res = market_service.ingest_cvm(db, ["KNRI11"], 2026, fetcher=lambda y, t: cvm.parse_zip(_cvm_zip(), t, "https://cvm.test"))
    assert res["saved"] == 2
    rows = client.get("/api/indicators", params={"ticker": "KNRI11"}).json()
    assert rows[0]["source"] == "CVM" and Dec(rows[0]["vp_per_share"]) == Dec("160.20")
    assert client.get("/api/assets").json()[0]["cnpj"] == "12.005.956/0001-65"


def test_cvm_failure_is_logged_not_raised(client):
    def boom(year, tickers):
        raise cvm.ProviderError("CVM HTTP 403")

    with SessionLocal() as db:
        res = market_service.ingest_cvm(db, ["KNRI11"], 2026, fetcher=boom)
    assert res["saved"] == 0 and "403" in res["warnings"][0]
    assert client.get("/api/market/logs").json()[0]["status"] == "ERROR"
