import io
import json
import zipfile
from datetime import date
from decimal import Decimal as Dec

import httpx
import pytest

from app.assistant import agent
from app.assistant.tools import DISPATCH, to_jsonable
from app.core.db import SessionLocal
from app.domain.screener import ScreenCriteria, UniverseFund, normalize_segment, screen
from app.market.providers import ListedFund, list_funds
from app.seed import seed_demo_market, seed_portfolio
from app.services import universe_service

TODAY = date(2026, 10, 11)


# ---------------- triagem (domínio)

def uf(t, close="100", vol="20000", **kw):
    base = dict(segment="Logística", net_worth=Dec("1000000000"), shareholders=50000, vp_per_share=Dec("105"))
    base.update(kw)
    return UniverseFund(t, kw.pop("sub_type", "fii") if "sub_type" in kw else "fii", Dec(close), Dec(vol), **{k: v for k, v in base.items() if k != "sub_type"})


def test_screen_filters_and_explains():
    universe = [
        uf("GOOD11"),
        uf("ILLQ11", vol="100"),  # R$ 10 mil
        uf("SMAL11", net_worth=Dec("50000000")),
        uf("PREM11", vp_per_share=Dec("80")),  # P/VP 1,25
        uf("CHEA11", vp_per_share=Dec("200")),  # P/VP 0,50
        uf("HELD11"),
        UniverseFund("AGRO11", "fi-agro", Dec("10"), Dec("500000")),
    ]
    r = screen(universe, ScreenCriteria(exclude={"HELD11"}))
    assert [x["ticker"] for x in r["shortlist"]] == ["GOOD11"]
    assert r["rejected_by_first_reason"] == {"liquidez baixa": 1, "patrimônio baixo": 1, "P/VP fora da faixa": 2, "já na carteira": 1, "tipo de fundo": 1}


def test_screen_segment_filter_ignores_accents_and_flags_missing_cvm():
    universe = [uf("LOG11", segment="Logística"), uf("SHO11", segment="Shoppings"), UniverseFund("NOC11", "fii", Dec("100"), Dec("20000"))]
    r = screen(universe, ScreenCriteria(segment="logist"))
    assert [x["ticker"] for x in r["shortlist"]] == ["LOG11"]
    r2 = screen(universe, ScreenCriteria())
    noc = next(x for x in r2["shortlist"] if x["ticker"] == "NOC11")
    assert "sem dados da CVM" in noc["notes"][0]
    assert screen(universe, ScreenCriteria(require_cvm=True))["passed"] == 2


def test_segment_normalization_matches_portfolio_labels():
    assert normalize_segment("Títulos e Val. Mob.") == "Recebíveis (CRI)"
    assert normalize_segment("Logística") == "Logística"
    assert normalize_segment("Algo novo") == "Algo novo"


# ---------------- BRAPI /quote/list (formato real de 14/10/2026)

def test_list_funds_paginates_and_keeps_only_fund_subtypes():
    pages = {
        "1": {"stocks": [{"stock": "BOVA11", "close": 200.9, "volume": 2049187, "type": "fund", "subType": "etf"},
                         {"stock": "MXRF11", "name": "MXRF11", "close": 9.34, "volume": 504562, "type": "fund", "subType": "fii", "subsector": "Logística"}],
              "hasNextPage": True},
        "2": {"stocks": [{"stock": "RURA11", "close": 9.0, "volume": 100000, "type": "fund", "subType": "fi-agro"}], "hasNextPage": False},
    }
    c = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=pages[req.url.params["page"]])))
    fs = list_funds("https://brapi.test/api", client=c)
    assert [(f.ticker, f.sub_type) for f in fs] == [("MXRF11", "fii"), ("RURA11", "fi-agro")]
    assert fs[0].close == Dec("9.34")


# ---------------- universo + CVM (ZIP sintético no layout pós-Res. 175)

def _cvm_zip():
    geral = ("CNPJ_Fundo_Classe;Data_Referencia;Nome_Fundo_Classe;Codigo_ISIN;Segmento_Atuacao;Mandato\n"
             "11.111.111/0001-11;2026-08-01;CSHG LOGISTICA;BRHGLGCTF004;Logística;Renda\n"
             "22.222.222/0001-22;2026-08-01;MAXI RENDA;BRMXRFCTF008;Títulos e Val. Mob.;Títulos e Valores Mobiliários\n")
    comp = ("CNPJ_Fundo_Classe;Data_Referencia;Patrimonio_Liquido;Cotas_Emitidas;Valor_Patrimonial_Cotas;Total_Numero_Cotistas\n"
            "11.111.111/0001-11;2026-07-01;5000000000;30000000;166.00;400000\n"
            "11.111.111/0001-11;2026-08-01;5100000000;30000000;170.00;410000\n"
            "22.222.222/0001-22;2026-08-01;4000000000;420000000;9.52;1200000\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("inf_mensal_fii_geral_2026.csv", geral.encode("latin-1"))
        z.writestr("inf_mensal_fii_complemento_2026.csv", comp.encode("latin-1"))
    return buf.getvalue()


def _refresh(db, cvm_ok=True):
    listed = [ListedFund("HGLG11", "HGLG11", "fii", Dec("160"), Dec("25000")), ListedFund("MXRF11", "MXRF11", "fii", Dec("9.34"), Dec("504562")),
              ListedFund("TINY11", "TINY11", "fii", Dec("50"), Dec("10"))]

    def cvm_fetch():
        if not cvm_ok:
            raise universe_service.ProviderError("CVM HTTP 403")
        return _cvm_zip(), "https://cvm.test/inf.zip"

    return universe_service.refresh_universe(db, "x", None, 2026, list_fetcher=lambda: listed, cvm_fetcher=cvm_fetch)


def test_universe_refresh_enriches_with_cvm_and_screens(client):
    with SessionLocal() as db:
        out = _refresh(db)
        assert out == {"listed": 3, "with_cvm": 2, "warnings": []}
        r = universe_service.run_screen(db, 1, ScreenCriteria())
    sl = {x["ticker"]: x for x in r["shortlist"]}
    assert set(sl) == {"HGLG11", "MXRF11"}
    assert sl["HGLG11"]["segment"] == "Logística" and sl["HGLG11"]["shareholders"] == 410000
    assert round(sl["HGLG11"]["pvp"], 3) == Dec("0.941")  # 160 / 170 (informe mais recente)
    assert sl["MXRF11"]["segment"] == "Recebíveis (CRI)"
    # pela API, com a carteira semeada, MXRF11 sai da triagem por já estar na carteira
    with SessionLocal() as db:
        seed_portfolio(db)
    api = client.get("/api/universe/screen").json()
    assert [x["ticker"] for x in api["shortlist"]] == ["HGLG11"]


def test_universe_refresh_survives_cvm_failure(client):
    with SessionLocal() as db:
        out = _refresh(db, cvm_ok=False)
    assert out["listed"] == 3 and out["with_cvm"] == 0 and "403" in out["warnings"][0]
    r = client.get("/api/universe/screen").json()
    assert any("Sem dados da CVM" in n for n in r["notes"])


def test_watchlist_toggle(client):
    with SessionLocal() as db:
        _refresh(db)
    assert client.post("/api/watchlist/HGLG11").json() == {"ticker": "HGLG11", "watchlist": True}
    a = next(x for x in client.get("/api/assets").json() if x["ticker"] == "HGLG11")
    assert a["is_active"] and a["segment"] == "Logística"
    assert client.delete("/api/watchlist/HGLG11").json()["watchlist"] is False


# ---------------- ferramentas do assistente

def _seed():
    with SessionLocal() as db:
        seed_portfolio(db)
        seed_demo_market(db, TODAY)


def test_tools_return_real_data_and_flag_demo(client):
    _seed()
    with SessionLocal() as db:
        port = to_jsonable(DISPATCH["get_portfolio"](db, 1, TODAY))
        budget = to_jsonable(DISPATCH["get_budget"](db, 1, TODAY))
        recs = to_jsonable(DISPATCH["get_recommendations"](db, 1, TODAY, profile="RENDA", budget=1500))
        sim = to_jsonable(DISPATCH["simulate"](db, 1, TODAY, monthly_contribution=300, target_monthly_income=500))
        need = to_jsonable(DISPATCH["simulate"](db, 1, TODAY, monthly_contribution=300, years=15, target_monthly_income=2000, solve_contribution=True))
    assert port["uses_demo_data"] and {p["ticker"] for p in port["positions"]} == {"KNCR11", "KNRI11", "MXRF11", "RURA11", "SNEL11", "VILG11", "XPML11"}
    assert budget["current_month"]["available"] == "300.00"
    assert recs["uses_demo_data"] and [p["ticker"] for p in recs["picks"]] == ["MXRF11", "SNEL11"]
    assert sim["time_to_target"]["reached"] and Dec(need["required_contribution"]["monthly_contribution"]) > 0
    json.dumps(port), json.dumps(recs)  # serializável para o modelo


# ---------------- loop do agente com OpenAI simulada

def _openai(script):
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        return httpx.Response(200, json=script[len(calls) - 1](body))

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_agent_calls_tools_and_returns_trace(client):
    _seed()
    script = [
        lambda b: {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_budget", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "get_recommendations", "arguments": json.dumps({"budget": 500})}}]}}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        lambda b: {"choices": [{"message": {"role": "assistant", "content": "Aviso: dados DEMO. Sugiro ... Fontes: get_budget, get_recommendations."}}],
                   "usage": {"prompt_tokens": 900, "completion_tokens": 80}, "model": "gpt-test"},
    ]
    http, calls = _openai(script)
    with SessionLocal() as db:
        out = agent.run(db, 1, TODAY, [{"role": "user", "content": "Onde aportar R$ 500?"}], api_key="k", model="m", base_url="https://oa.test/v1", client=http)
    assert [t["tool"] for t in out["tools"]] == ["get_budget", "get_recommendations"]
    assert all(t["ok"] for t in out["tools"]) and out["tools"][1]["uses_demo_data"]
    assert out["usage"] == {"prompt_tokens": 1000, "completion_tokens": 100}
    # o modelo recebeu as regras e o resultado das ferramentas
    assert "Nunca estime de memória" in calls[0]["messages"][0]["content"]
    tool_msgs = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and '"budget": "500.00"' in tool_msgs[1]["content"]
    assert {t["function"]["name"] for t in calls[0]["tools"]} >= {"screen_new_funds", "evaluate_new_funds"}


def test_agent_reports_tool_errors_back_to_model(client):
    script = [
        lambda b: {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "inexistente", "arguments": "{}"}}]}}]},
        lambda b: {"choices": [{"message": {"role": "assistant", "content": "Não consegui consultar."}}]},
    ]
    http, calls = _openai(script)
    with SessionLocal() as db:
        out = agent.run(db, 1, TODAY, [{"role": "user", "content": "oi"}], api_key="k", model="m", base_url="https://oa.test/v1", client=http)
    assert out["tools"][0]["ok"] is False and "desconhecida" in calls[1]["messages"][-1]["content"]


def test_agent_without_key_is_explicit(client):
    r = client.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "oi"}]})
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]
    assert client.get("/api/assistant/status").json()["configured"] is False


def test_agent_stops_after_max_rounds(client):
    loop = lambda b: {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [  # noqa: E731
        {"id": "x", "type": "function", "function": {"name": "get_alerts", "arguments": "{}"}}]}}]}
    http, calls = _openai([loop] * 10)
    with SessionLocal() as db:
        out = agent.run(db, 1, TODAY, [{"role": "user", "content": "?"}], api_key="k", model="m", base_url="https://oa.test/v1", client=http)
    assert len(calls) == agent.MAX_ROUNDS and "limite de consultas" in out["answer"]
