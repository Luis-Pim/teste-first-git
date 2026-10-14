from decimal import Decimal as Dec

import httpx

from app.core.db import SessionLocal
from app.market.providers import BcbSgsClient, BrapiProvider
from app.seed import seed_demo_market, seed_portfolio
from app.services import market_service

TODAY = "2026-10-11"
Q = {"as_of": TODAY}


def post(c, url, json, **kw):
    r = c.post(url, json=json, params=Q, **kw)
    assert r.status_code in (200, 201), r.text
    return r.json()


def test_buy_sell_flow_and_summary(client):
    post(client, "/api/transactions", {"type": "BUY", "trade_date": "2026-09-01", "ticker": "KNRI11", "quantity": 10, "unit_price": "100", "fees": "1"})
    post(client, "/api/transactions", {"type": "SELL", "trade_date": "2026-09-10", "ticker": "KNRI11", "quantity": 4, "unit_price": "110"})
    post(client, "/api/quotes", {"ticker": "KNRI11", "quote_date": "2026-10-10", "close": "120"})
    pos = client.get("/api/portfolio/positions", params=Q).json()["positions"][0]
    assert Dec(pos["quantity"]) == 6
    assert Dec(pos["avg_price"]) == Dec("100.1")
    assert Dec(pos["market_value"]) == Dec("720.00")
    assert Dec(pos["realized_pnl"]) == Dec("39.60")
    s = client.get("/api/portfolio/summary", params=Q).json()
    assert Dec(s["invested_total"]) == Dec("600.60") and Dec(s["market_value"]) == Dec("720.00")
    assert s["total_return"] is not None


def test_oversell_rejected_and_not_persisted(client):
    post(client, "/api/transactions", {"type": "BUY", "trade_date": "2026-09-01", "ticker": "MXRF11", "quantity": 5, "unit_price": "10"})
    r = client.post("/api/transactions", json={"type": "SELL", "trade_date": "2026-09-02", "ticker": "MXRF11", "quantity": 6, "unit_price": "10"})
    assert r.status_code == 422 and "excede" in r.text
    assert len(client.get("/api/transactions").json()) == 1


def test_deleting_buy_that_backs_a_sale_is_rejected(client):
    b = post(client, "/api/transactions", {"type": "BUY", "trade_date": "2026-09-01", "ticker": "MXRF11", "quantity": 5, "unit_price": "10"})
    post(client, "/api/transactions", {"type": "SELL", "trade_date": "2026-09-02", "ticker": "MXRF11", "quantity": 5, "unit_price": "10"})
    assert client.delete(f"/api/transactions/{b['id']}").status_code == 422


def test_validation_messages(client):
    r = client.post("/api/transactions", json={"type": "BUY", "trade_date": "2026-09-01", "ticker": "X11", "quantity": 1})
    assert r.status_code == 422


def test_seed_pending_subscription_and_confirmation(client):
    with SessionLocal() as db:
        seed_portfolio(db)
    pos = {p["ticker"]: p for p in client.get("/api/portfolio/positions", params=Q).json()["positions"]}
    assert Dec(pos["KNCR11"]["quantity"]) == 17 and Dec(pos["KNCR11"]["pending_quantity"]) == 10
    assert pos["KNCR11"]["cost_known"] is False
    assert Dec(pos["MXRF11"]["planned_quantity"]) == 100 and Dec(pos["MXRF11"]["quantity"]) == 50
    s = client.get("/api/portfolio/summary", params=Q).json()
    assert Dec(s["invested_pending"]) == Dec("1046.80") and s["total_return"] is None and len(s["cost_unknown"]) == 7
    alerts = {a["type"] for a in client.get("/api/alerts", params=Q).json()}
    assert {"SUBSCRICAO_PENDENTE", "ORDEM_PLANEJADA", "PRECO_MEDIO_AUSENTE", "SEM_COTACAO"} <= alerts

    pending = client.get("/api/transactions", params={"status": "PENDING"}).json()[0]
    r = client.post(f"/api/transactions/{pending['id']}/confirm", json={"settlement_date": "2026-10-20"})
    assert r.status_code == 200 and r.json()["status"] == "CONFIRMED"
    pos = {p["ticker"]: p for p in client.get("/api/portfolio/positions", params=Q).json()["positions"]}
    assert Dec(pos["KNCR11"]["quantity"]) == 27


def test_dividend_eligibility_and_budget(client):
    client.put("/api/planning/strategy", params=Q, json={
        "monthly_contribution": "300", "reinvest_pct": "1", "annual_increase_pct": "0", "start_month": "2026-10-01",
        "initial_cash": "40", "extra_contributions": [{"month": 6, "amount": "5000"}, {"month": 12, "amount": "5000"}]})
    post(client, "/api/transactions", {"type": "BUY", "trade_date": "2026-09-01", "ticker": "KNRI11", "quantity": 10, "unit_price": "100"})
    post(client, "/api/transactions", {"type": "BUY", "trade_date": "2026-10-01", "ticker": "KNRI11", "quantity": 5, "unit_price": "100"})
    ann = post(client, "/api/dividends/announcements", {"ticker": "KNRI11", "amount_per_share": "1.00", "com_date": "2026-09-30", "payment_date": "2026-10-14"})
    inc = client.get("/api/dividends/income", params=Q).json()
    assert Dec(inc["confirmed"][0]["eligible_quantity"]) == 10 and Dec(inc["confirmed"][0]["amount"]) == Dec("10.00")
    # pagamento registrado vira RECEBIDA e alimenta o orçamento
    post(client, "/api/transactions", {"type": "DIVIDEND", "trade_date": "2026-10-14", "ticker": "KNRI11", "total_amount": "10.00", "announcement_id": ann["id"]})
    inc = client.get("/api/dividends/income", params=Q).json()
    assert inc["confirmed"] == [] and len(inc["received"]) == 1
    led = client.get("/api/planning/ledger", params={**Q, "months_ahead": 0}).json()
    cur = led["current"]
    # 40 + 300 (planejado) + 10 (dividendo) − 500 (compra de 01/10) = −150
    assert Dec(cur["available"]) == Dec("350.00") and Dec(cur["carry_out"]) == Dec("-150.00")


def test_simulator_endpoints(client):
    body = {"initial_value": "10000", "monthly_contribution": "300", "years": 10}
    r = post(client, "/api/simulator/run", body)
    assert set(r["scenarios"]) == {"conservador", "base", "otimista"}
    base = r["scenarios"]["base"]["final"]
    assert Dec(r["scenarios"]["otimista"]["final"]["patrimonio"]) > Dec(base["patrimonio"]) > Dec(r["scenarios"]["conservador"]["final"]["patrimonio"])
    t = post(client, "/api/simulator/goal-time", {**body, "target_monthly_income": "500"})
    assert t["reached"]
    g = post(client, "/api/simulator/goal-contribution", {**body, "target_monthly_income": "2000", "years": 15})
    assert Dec(g["monthly_contribution"]) > 0
    e = post(client, "/api/simulator/extras-impact", {**body, "compare_extras": [{"month": 6, "amount": "5000"}, {"month": 12, "amount": "5000"}]})
    assert Dec(e["delta_patrimonio"]) > 0


IMPORT_CSV = (
    "data;tipo;status;ticker;quantidade;preco_unitario;taxas;valor_total;proporcao_de;proporcao_para;observacao\n"
    "01/09/2026;COMPRA;CONFIRMADA;VILG11;3;90,00;0;;;;\n"
    "01/09/2026;COMPRA;CONFIRMADA;VILG11;3;90,00;0;;;;\n"
    "14/09/2026;RENDIMENTO;CONFIRMADA;VILG11;6;;;4,20;;;\n"
)


def test_import_preview_commit_and_dedup(client):
    files = {"file": ("ops.csv", IMPORT_CSV.encode(), "text/csv")}
    prev = client.post("/api/imports/preview", files=files).json()
    assert prev["counts"] == {"NOVA": 3}
    assert prev["reconciliation"] == [{"ticker": "VILG11", "before": "0", "after": "6", "delta": "6"}]
    res = client.post("/api/imports/commit", files={"file": ("ops.csv", IMPORT_CSV.encode(), "text/csv")}).json()
    assert res["imported"] == 3
    again = client.post("/api/imports/preview", files={"file": ("ops.csv", IMPORT_CSV.encode(), "text/csv")}).json()
    assert again["counts"] == {"DUPLICADA": 3}
    res2 = client.post("/api/imports/commit", files={"file": ("ops.csv", IMPORT_CSV.encode(), "text/csv")}).json()
    assert res2["imported"] == 0


def test_import_flags_possible_duplicate_of_manual_entry(client):
    post(client, "/api/transactions", {"type": "BUY", "trade_date": "2026-09-01", "ticker": "VILG11", "quantity": 3, "unit_price": "90"})
    prev = client.post("/api/imports/preview", files={"file": ("ops.csv", IMPORT_CSV.encode(), "text/csv")}).json()
    assert prev["counts"] == {"POSSIVEL_DUPLICATA": 1, "NOVA": 2}


def test_dashboard_with_demo_data(client):
    with SessionLocal() as db:
        seed_portfolio(db)
        from datetime import date
        seed_demo_market(db, date(2026, 10, 11))
    d = client.get("/api/dashboard", params=Q).json()
    assert d["summary"]["price_sources"] == ["DEMO"]
    assert d["summary"]["market_value_complete"] is True
    assert d["allocation"]["basis"] == "MERCADO"
    assert Dec(d["budget"]["available"]) == Dec("300.00")
    assert any(u["ticker"] == "KNCR11" for u in d["upcoming_payments"])  # anúncio DEMO pago em 14/10
    det = client.get("/api/assets/KNRI11/detail", params=Q).json()
    assert det["indicators"]["available"] is True and det["indicators"]["rows"][0]["source"] == "DEMO"
    assert Dec(det["dividend_stats"]["sum_12m"]) == Dec("12")


# ---------------------------------------------------------------- adaptadores de mercado (HTTP simulado)

def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_brapi_adapter_parses_quotes_and_flags_missing_dividends(client):
    def handler(req: httpx.Request):
        assert req.headers["Authorization"] == "Bearer tkn"
        t = req.url.path.rsplit("/", 1)[-1]
        if t == "XXXX11":
            return httpx.Response(404, json={"error": True, "message": "not found"})
        return httpx.Response(200, json={"results": [{"symbol": t, "regularMarketPrice": 101.5, "regularMarketTime": "2026-10-10T20:07:00.000Z"}]})

    p = BrapiProvider("tkn", "https://brapi.test/api", client=_mock(handler))
    rep = p.fetch(["KNRI11", "XXXX11"], with_dividends=True)
    assert rep.quotes[0].close == Dec("101.5")
    assert "Proventos indisponíveis" in rep.missing["KNRI11"] and "HTTP 404" in rep.missing["XXXX11"]
    assert BrapiProvider(None, "x").fetch(["KNRI11"]).missing["KNRI11"].startswith("BRAPI sem token")


def test_market_refresh_uses_cache_and_logs(client):
    post(client, "/api/transactions", {"type": "BUY", "trade_date": "2026-09-01", "ticker": "KNRI11", "quantity": 1, "unit_price": "100"})
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json={"results": [{"regularMarketPrice": 150.25, "regularMarketTime": "2026-10-10T20:00:00Z",
                                                      "dividendsData": {"cashDividends": [{"rate": 1.0, "paymentDate": "2026-10-14T00:00:00.000Z", "lastDatePrior": "2026-09-30T00:00:00.000Z", "label": "RENDIMENTO"}]}}]})

    p = BrapiProvider("tkn", "https://brapi.test/api", client=_mock(handler))
    with SessionLocal() as db:
        r1 = market_service.refresh(db, p, ["KNRI11"], cache_minutes=15, with_dividends=True)
        r2 = market_service.refresh(db, p, ["KNRI11"], cache_minutes=15)
    assert r1["quotes_saved"] == 1 and r1["dividends_saved"] == 1
    assert r2["from_cache"] == ["KNRI11"] and len(calls) == 1
    logs = client.get("/api/market/logs").json()
    assert len(logs) == 2 and logs[1]["status"] == "OK"


def test_bcb_client_parses_series():
    c = BcbSgsClient("https://bcb.test", client=_mock(lambda req: httpx.Response(200, json=[{"data": "01/10/2026", "valor": "0,055131"}])))
    from datetime import date
    assert c.fetch("CDI", date(2026, 10, 1), date(2026, 10, 1)) == [(date(2026, 10, 1), Dec("0.055131"))]
