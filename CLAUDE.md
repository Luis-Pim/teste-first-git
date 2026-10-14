# Contexto para o Claude — projeto InvestAI

Este repositório tem dois projetos. A branch `investai` contém o **InvestAI** na pasta `investai/`. O restante (app.py, arquivos `tormenta_*`) é um projeto antigo de RPG e não deve ser alterado.

## Quem é o usuário e como trabalhar

- Conversa em **português do Brasil**. Uso **pessoal** (não é produto comercial).
- Investidor em FIIs com foco em renda passiva: aporte mínimo de R$ 300/mês, reinveste 100% dos dividendos, pelo menos dois aportes extras por ano (valores ainda a definir).
- Pode commitar e fazer push **na branch `investai`**. Não abrir PR nem mexer na `main` sem pedido.
- Nunca pedir chaves no chat. Chaves ficam nas variáveis de ambiente do ambiente de nuvem.
- Regras do produto que o usuário pediu e que devem continuar valendo:
  - valores monetários sempre em `Decimal`; a API serializa Decimal como string;
  - nunca inventar cotações, proventos ou indicadores; dado ausente é sinalizado;
  - dado fictício só com fonte `DEMO` e selo na interface;
  - posição sempre derivada das operações (nunca saldo ÷ preço médio);
  - não executar ordens reais; não criar infraestrutura AWS paga sem estimativa e autorização.

## Carteira do usuário (cadastrada pelo seed)

| Ativo | Cotas | Observação |
|---|---|---|
| KNCR11 | 17 | + 10 cotas PENDENTES do direito de preferência da 13ª emissão (≈ R$ 1.046,80, aguardando conversão) |
| KNRI11 | 8 | |
| MXRF11 | 50 | + compra PLANEJADA de 100 cotas (não executada) |
| RURA11 | 26 | |
| SNEL11 | 24 | |
| VILG11 | 5 | |
| XPML11 | 17 | |

Preços médios **não foram informados** (saldo inicial sem custo; P/L e YoC aparecem como "—"). Data do saldo inicial: 30/09/2026 (aproximada). Estratégia começa em 10/2026.

## Estado atual (14/10/2026)

- **Fase 1 (MVP)** concluída: carteira, movimentações (13 tipos, status confirmada/pendente/planejada), dividendos por data-com (recebida/confirmada/estimada), planejamento com saldo residual, simulador, importação CSV/Excel B3 com deduplicação, alertas.
- **Fase 2** concluída: motor de recomendação (`backend/app/domain/recommendation.py`, 7 fatores, pesos por perfil, vetos de "não comprar", alocação com limites 25% por fundo / 40% por segmento); tabela `fund_indicators`; adaptadores BRAPI e CVM.
- **Fase 3 (parcial)**: triagem de fundos novos (`domain/screener.py`, `services/universe_service.py`) e Assistente com OpenAI (`backend/app/assistant/`, 9 ferramentas, tela `/assistente`).
- 77 testes passando (`cd investai/backend && pytest -q`).
- Demo estática publicada (artifact): https://claude.ai/artifact/ACY8Rjj1VULX8BdP3hjo54

## Fatos verificados sobre as fontes

- BRAPI `/quote/{ticker}`: formato validado contra a resposta real (regularMarketVolume, historicalDataPrice com epoch, dividendsData.cashDividends com lastDatePrior=data-com, exDate, paymentDate). Datas vêm em UTC 03:00 = meia-noite de Brasília. Cotação de FII **exige token**.
- BRAPI `/quote/list?type=fund`: responde **sem token**; 385 fundos (331 fii, 37 fi-agro, 17 fi-infra). O volume é do pregão em andamento. O **setor/subsetor da BRAPI não é confiável** para FIIs (MXRF11 aparece como Logística) → segmento vem da CVM.
- CVM informe mensal (`inf_mensal_fii_{ano}.zip`): adaptador escrito e testado só com ZIP sintético. **Pendente validar** nomes de colunas contra o arquivo real.
- OpenAI: modelo padrão `gpt-4.1-mini` em `INVESTAI_OPENAI_MODEL`. **Pendente confirmar** que existe na conta (`GET /v1/models`).

## Próximos passos (nesta ordem)

1. Preparar o ambiente (abaixo) e rodar `cd investai/backend && .venv/bin/python -m app.diagnostico`: mostra, sem imprimir valores, se `INVESTAI_BRAPI_TOKEN` e `OPENAI_API_KEY` existem, se `brapi.dev`, `api.openai.com`, `dados.cvm.gov.br` e `api.bcb.gov.br` respondem e (com a chave) se o modelo da OpenAI existe na conta.
   - Sessão de 14/10/2026: as duas chaves estavam AUSENTES e o proxy de rede bloqueou os quatro domínios (403). O usuário precisa liberá-los em Network access (Allowed domains) e cadastrar as chaves no ambiente de nuvem antes dos passos 2–5.
2. Confirmar o modelo da OpenAI e ajustar o padrão se necessário.
3. Baixar o informe real da CVM e validar `app/market/cvm.py` (`parse_zip` e `parse_universe`); ajustar colunas e testes.
4. Rodar `python -m app.seed --clear-demo`, depois Configurações → Atualizar cotações + histórico 12 meses + proventos (BRAPI) e Importar VP da CVM, para trocar DEMO por dados reais.
5. Rodar o motor e a triagem com dados reais e revisar os resultados com o usuário; testar o Assistente com perguntas reais.
6. Pedir ao usuário os preços médios e os valores dos dois aportes extras.
7. Depois: job agendado de atualização, CDI/IPCA/IFIX (BCB) e comparação, RAG de relatórios gerenciais, alertas inteligentes.

## Preparar o ambiente numa sessão nova

```bash
git checkout investai   # se a sessão abrir em outra branch
service postgresql start
su postgres -c "psql -c \"CREATE USER investai WITH PASSWORD 'investai' CREATEDB;\" -c \"CREATE DATABASE investai OWNER investai;\" -c \"CREATE DATABASE investai_test OWNER investai;\""
cd investai/backend && python3 -m venv .venv && .venv/bin/pip install -q -r requirements-dev.txt
.venv/bin/alembic upgrade head && .venv/bin/python -m app.seed --demo-market && .venv/bin/python -m pytest -q
cd ../frontend && npm install --no-audit --no-fund && npx tsc --noEmit
# rodar: backend `.venv/bin/uvicorn app.main:app --port 8000`; frontend `npx next build && npx next start -p 3000`
```

Cuidado ao matar processos: `pkill -f "next start"` também mata o próprio shell. Encontre o PID primeiro e mate pelo número.
