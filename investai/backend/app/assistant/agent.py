"""Assistente InvestAI: loop de chamadas de ferramentas sobre a API Chat Completions da OpenAI.

O modelo só sabe o que as ferramentas devolvem. Cada resposta traz o rastro das ferramentas usadas,
para o usuário conferir de onde veio cada número.
"""

from __future__ import annotations

import json
from datetime import date

import httpx
from sqlalchemy.orm import Session

from app.assistant.tools import DISPATCH, TOOLS, to_jsonable

MAX_ROUNDS = 6

SYSTEM_PROMPT = """Você é o assistente financeiro do InvestAI, para uso pessoal de um investidor brasileiro em fundos imobiliários.
Data de referência: {today}.

Regras obrigatórias:
1. Todo número que você citar (cotação, provento, P/VP, saldo, cotas, renda) precisa ter vindo de uma ferramenta nesta conversa. Se não veio, diga que não tem o dado. Nunca estime de memória.
2. Antes de responder sobre carteira, orçamento, proventos ou recomendações, chame a ferramenta correspondente.
3. Para sugerir fundos que o usuário ainda não tem: chame screen_new_funds e depois evaluate_new_funds com os finalistas. Só sugira fundos que o motor considerou elegíveis; explique os vetos dos demais quando forem relevantes.
4. Nunca recomende só pelo maior dividend yield. Explique o porquê com os fatores do motor e cite os principais riscos.
5. Se uma ferramenta indicar uses_demo_data = true ou fonte DEMO, avise no início da resposta que os números são ilustrativos.
6. Termine com uma linha "Fontes:" listando as ferramentas e fontes de dados usadas, e as limitações relevantes (dados ausentes, cobertura baixa).
7. Você não executa ordens de compra ou venda. Projeções são hipotéticas.
Responda em português do Brasil, de forma direta, com valores em R$ no formato brasileiro."""


class AssistantUnavailable(RuntimeError):
    pass


def _tool_specs():
    return [{"type": "function", "function": t} for t in TOOLS]


def run(
    db: Session, pid: int, today: date, messages: list[dict], *, api_key: str | None, model: str, base_url: str,
    client: httpx.Client | None = None,
) -> dict:
    if not api_key:
        raise AssistantUnavailable("Chave da OpenAI não configurada (OPENAI_API_KEY).")
    client = client or httpx.Client(timeout=90.0)
    convo = [{"role": "system", "content": SYSTEM_PROMPT.format(today=today.strftime("%d/%m/%Y"))}]
    convo += [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant")]
    trace: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for _ in range(MAX_ROUNDS):
        try:
            r = client.post(f"{base_url.rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {api_key}"},
                            json={"model": model, "messages": convo, "tools": _tool_specs(), "tool_choice": "auto"})
        except httpx.HTTPError as e:
            raise AssistantUnavailable(f"OpenAI indisponível: {e.__class__.__name__}") from e
        if r.status_code != 200:
            raise AssistantUnavailable(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        for k in usage:
            usage[k] += (body.get("usage") or {}).get(k, 0)
        msg = body["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"answer": msg.get("content") or "", "tools": trace, "usage": usage, "model": body.get("model", model), "reference_date": today}
        convo.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = DISPATCH.get(name)
            if fn is None:
                result, ok = {"error": f"ferramenta desconhecida: {name}"}, False
            else:
                try:
                    result, ok = to_jsonable(fn(db, pid, today, **args)), True
                except Exception as e:  # noqa: BLE001 — o erro volta ao modelo, que deve dizê-lo ao usuário
                    db.rollback()
                    result, ok = {"error": f"{e.__class__.__name__}: {e}"}, False
            trace.append({"tool": name, "arguments": args, "ok": ok, "uses_demo_data": bool(isinstance(result, dict) and result.get("uses_demo_data"))})
            convo.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)[:60000]})
    return {"answer": "Não consegui concluir a análise dentro do limite de consultas. Tente uma pergunta mais específica.",
            "tools": trace, "usage": usage, "model": model, "reference_date": today}
