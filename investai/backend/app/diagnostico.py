"""Diagnóstico do ambiente: chaves presentes e fontes externas acessíveis.

Uso: `python -m app.diagnostico`. Nunca imprime o valor das chaves.
"""

import sys

import httpx

from app.core.config import get_settings

CVM_URL = "https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/"


def _probe(client: httpx.Client, nome: str, url: str, aceitar_401: bool = False) -> bool:
    try:
        resp = client.get(url)
    except httpx.HTTPError as exc:
        print(f"  {nome:<8} FALHOU  ({type(exc).__name__}: {exc})")
        return False
    ok = resp.status_code < 400 or (aceitar_401 and resp.status_code == 401)
    print(f"  {nome:<8} {'OK    ' if ok else 'ERRO  '} (HTTP {resp.status_code})")
    return ok


def main() -> int:
    s = get_settings()
    print("Chaves:")
    for nome, valor in [("INVESTAI_BRAPI_TOKEN", s.brapi_token), ("OPENAI_API_KEY", s.openai_api_key)]:
        print(f"  {nome:<21} {'definida' if valor else 'AUSENTE'}")

    print("Fontes:")
    tudo_ok = True
    with httpx.Client(timeout=s.http_timeout_seconds, follow_redirects=True) as client:
        tudo_ok &= _probe(client, "BRAPI", f"{s.brapi_base_url}/quote/list?type=fund&limit=1")
        tudo_ok &= _probe(client, "CVM", CVM_URL)
        tudo_ok &= _probe(client, "BCB", f"{s.bcb_base_url}/bcdata.sgs.12/dados/ultimos/1?formato=json")
        if s.openai_api_key:
            try:
                resp = client.get(f"{s.openai_base_url}/models", headers={"Authorization": f"Bearer {s.openai_api_key}"})
                resp.raise_for_status()
                modelos = {m["id"] for m in resp.json().get("data", [])}
                disponivel = s.openai_model in modelos
                print(f"  OpenAI   OK     (modelo {s.openai_model}: {'disponível' if disponivel else 'NÃO disponível'})")
                tudo_ok &= disponivel
            except httpx.HTTPError as exc:
                print(f"  OpenAI   FALHOU  ({type(exc).__name__})")
                tudo_ok = False
        else:
            tudo_ok &= _probe(client, "OpenAI", f"{s.openai_base_url}/models", aceitar_401=True)
    return 0 if tudo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
