# InvestAI — Arquitetura, decisões e plano

## 1. Arquitetura proposta

```
 Navegador ──HTTPS──▶ Next.js (UI)  ──JSON──▶ FastAPI ──▶ PostgreSQL
                                                │
                                                ├── domain/   regras financeiras puras (Decimal)
                                                ├── services/ composição de visões, alertas
                                                ├── market/   adaptadores de dados (Manual, BRAPI, BCB, CVM…)
                                                └── jobs (Fase 2): atualização agendada + log de execução
                                                          │
                                   Fase 3: Advisor (LLM) ─┘ usa os MESMOS endpoints como ferramentas + RAG de relatórios
```

Princípios:

- **Domínio puro e testável.** Todo cálculo (posição, preço médio, elegibilidade na data-com, orçamento, projeção) está em `backend/app/domain`, sem banco nem rede. A API apenas carrega dados e chama o domínio.
- **Posição derivada, nunca armazenada.** Saldo de cotas e custo médio são reconstruídos a partir das operações a cada consulta. Não existe "saldo ÷ preço médio". Editar ou excluir uma operação antiga recalcula tudo; alterações que gerariam saldo negativo são rejeitadas (HTTP 422).
- **Decimal ponta a ponta.** Colunas `NUMERIC`, cálculos em `Decimal`, e a API serializa valores como **string** (o encoder padrão do FastAPI converteria para float — isso foi sobrescrito em `main.py`).
- **Proveniência explícita.** Cotações e proventos carregam `source` (MANUAL, BRAPI, DEMO…); toda chamada externa gera um `data_fetch_logs`. Dados DEMO aparecem com selo amarelo na UI.
- **Ausência é informação.** Sem preço médio → P/L "—" e aviso; sem cotação → valor de mercado parcial e aviso; sem indicadores → "indisponível" com o motivo.

## 2. Modelo de dados

| Tabela | Papel |
|---|---|
| `users`, `portfolios` | Multiusuário/multicarteira preparado (MVP usa a carteira 1) |
| `assets` | Ticker, nome, classe (FII, ação, ETF, Tesouro, renda fixa, direito), segmento, gestora, CNPJ |
| `transactions` | Toda movimentação: tipo, **status** (CONFIRMED / PENDING / PLANNED), datas, qtd, preço, taxas, valor, proporção (desdobro/grupamento), vínculo com anúncio, origem, **hash de importação** (único por carteira) |
| `price_quotes` | Cotação por ativo/data/fonte |
| `dividend_announcements` | Anúncio por ativo: valor/cota, data-com, data-ex, pagamento, tipo (rendimento/amortização), fonte, URL |
| `strategy_configs` | Aporte mensal, % reinvestimento, aumento anual, extras (mês/valor), início, caixa inicial, perfil, limites de concentração |
| `benchmark_points` | Séries CDI / IPCA / IFIX (Fase 2) |
| `import_batches` | Rastreio de cada importação (totais, duplicadas, erros) |
| `data_fetch_logs` | Log de toda consulta a fonte externa |

Tipos de operação: saldo inicial, compra, venda, subscrição, direito de preferência, sobras, amortização, desdobramento, grupamento, rendimento recebido, taxa, aporte e retirada de dinheiro.

Regras-chave:

| Situação | Regra |
|---|---|
| Compra / subscrição | custo += qtd × preço + taxas |
| Venda | baixa custo pelo preço médio vigente; preço médio não muda; gera L/P realizado |
| Amortização | reduz o custo total (devolução de capital), quantidade inalterada |
| Desdobro / grupamento | qtd × (para ÷ de); custo total inalterado |
| PENDING (ex.: recibo de subscrição) | conta como capital investido e saída de caixa, mas fica fora das cotas confirmadas e da elegibilidade a proventos até ser confirmada |
| PLANNED | não afeta posição nem caixa; aparece como "comprometido" no orçamento |
| Elegibilidade | cotas **confirmadas** com data de negócio ≤ data-com |

## 3. Fontes de dados e limitações

| Fonte | O que oferece | Limitações / status |
|---|---|---|
| **Manual** | Cotações e proventos digitados | Sempre disponível. Ativo no MVP. |
| **BRAPI** | Cotação (`/quote/{ticker}`); proventos via `dividends=true` | Exige token; plano gratuito com cota mensal e módulos restritos. **Não presumimos** proventos/histórico de FIIs no plano gratuito: ausência do bloco é reportada como "indisponível". Adaptador pronto e testado com respostas simuladas. |
| **BCB SGS** | CDI (série 12), Selic (11), IPCA (433) | Público, sem token. Cliente pronto; persistência e comparação na Fase 2. |
| **IBGE SIDRA** | IPCA oficial (tabela 1737) | Fase 2. |
| **CVM Dados Abertos** | Informe mensal de FII: PL, VP/cota, cotistas, emissões | CSVs mensais, com defasagem. Base para P/VP. Fase 2. |
| **B3** | IFIX, eventos corporativos | Termos de uso restritivos para redistribuição; avaliar licença. Fase 2. |
| **Relatórios gerenciais** | Vacância, inadimplência, alavancagem, contratos | PDFs não padronizados → extração + RAG. Fase 3. |

> Observação: neste ambiente de desenvolvimento a rede de saída bloqueia brapi.dev e api.bcb.gov.br, por isso os adaptadores foram validados com HTTP simulado (`httpx.MockTransport`). Ao rodar localmente, configure `INVESTAI_MARKET_PROVIDER=brapi` e o token para testar contra a API real e confirmar o que o seu plano retorna.

Cache e controle de requisições: cotações buscadas há menos de `INVESTAI_QUOTE_CACHE_MINUTES` (15) não são refeitas; uma requisição por ativo (compatível com o plano gratuito); toda execução é registrada em `data_fetch_logs` (visível em Configurações).

## 4. Plano de implementação

| Fase | Entregas | Status |
|---|---|---|
| **1 — MVP** | Interface web, carteira, banco + migrações, movimentações (todos os tipos), posição e preço médio, dashboard, dividendos registrados + anúncios + calendário (3 categorias), planejamento mensal com saldo residual, simulador (3 cenários + metas), importação CSV/Excel com deduplicação e conciliação, alertas básicos, adaptadores BRAPI/BCB | ✅ |
| **2 — Recomendação + dados de mercado** | Motor de recomendações explicável (3 perfis, vetos de "não comprar", alocação respeitando limites); indicadores por fundo com fonte (VP/cota, vacância, inadimplência, alavancagem); BRAPI com volume, histórico diário e data-ex (formato validado contra a resposta real); informe mensal da CVM → P/VP; tela de Recomendações e indicadores na tela do fundo | ✅ motor e adaptadores · ⏳ validar CVM e BRAPI com token real |
| **2b — Mercado** | Job agendado para cotações, proventos e CVM; séries CDI/IPCA/IFIX e comparação; TWR | próximo |
| **3 — Inteligência** | ✅ Triagem de fundos novos (universo BRAPI + CVM) e Assistente com ferramentas sobre a API (OpenAI). ⏳ RAG de relatórios gerenciais; alertas inteligentes e e-mail | em andamento |
| **4 — Evolução** | Ações, ETFs, Tesouro, CDB; múltiplas carteiras; consolidação; integração com corretoras quando autorizada | — |

## 5. Estimativa de custos (AWS sa-east-1)

Valores **aproximados**, em USD/mês, para conferência na AWS Pricing Calculator antes de qualquer provisionamento. **Nenhum recurso AWS foi criado** — isso depende da sua autorização.

| Opção | Componentes | Custo estimado |
|---|---|---|
| **A — Mínima (recomendada para o MVP)** | 1 instância Lightsail (2 GB) rodando o `docker-compose` + snapshots automáticos + HTTPS via Caddy/Let's Encrypt | ~US$ 12–15 |
| **B — Gerenciada** | App Runner ou ECS Fargate (API + web), RDS PostgreSQL db.t4g.micro (20 GB, backup automático), Secrets Manager, CloudWatch | ~US$ 45–80 |
| Comuns | Route 53 (US$ 0,50/zona), ACM (grátis), Secrets Manager (US$ 0,40/segredo), AWS Budgets com alerta (grátis até 2 orçamentos) | ~US$ 1–3 |
| IA (Fase 3) | API de LLM sob demanda; uso pessoal tipicamente poucos dólares/mês | variável, com teto configurável |

Recomendação: começar pela opção A com alerta de orçamento em US$ 20; migrar para B quando houver mais usuários.

## 6. Critérios de aceitação — Fase 1

| # | Critério | Como verificar |
|---|---|---|
| 1 | Cotas vêm do replay das operações | `test_portfolio_engine.py` |
| 2 | Preço médio ponderado com taxas; venda parcial não altera PM e gera L/P realizado | testes + tela Carteira |
| 3 | Venda acima do saldo (inclusive por exclusão/edição retroativa) é rejeitada | `test_oversell_*`, `test_deleting_buy_*` |
| 4 | Subscrição pendente separada; confirmação a incorpora | `test_pending_subscription_*`, `test_seed_pending_*` |
| 5 | Ordem planejada não altera posição nem caixa | `test_planned_orders_*` |
| 6 | Valor investido ≠ valor de mercado; ausências sinalizadas | `/api/portfolio/summary` (`cost_unknown`, `missing_prices`) |
| 7 | Proventos usam cotas elegíveis na data-com; recebido não é contado como confirmado | `test_confirmed_income_*`, `test_received_*` |
| 8 | Renda separada em Recebida / Confirmada / Estimada, com premissa declarada | `/api/dividends/income` |
| 9 | Orçamento: 300 + 85 + 40 = 425; saldo residual passa para o mês seguinte | `test_budget_*` |
| 10 | Simulador 1–30 anos, 3 cenários, valores reais, metas (tempo até R$ 500, aporte para R$ 2.000, impacto de extras) | `test_goal_solvers`, tela Simulador |
| 11 | Importação: detecção de formato, duplicadas por hash, possível duplicata de lançamento manual, conciliação | `test_import_*` |
| 12 | Nenhum dado fictício sem identificação | dados DEMO com `source=DEMO` e selo na UI |
| 13 | Nenhuma chave no código | `.env` / variáveis `INVESTAI_*` |

## 7. Motor de recomendação (Fase 2)

Código: `backend/app/domain/recommendation.py` (puro, testado em `tests/test_recommendation.py`).

| Fator | Nota 0–100 | Conservador | Equilibrado | Renda |
|---|---|---|---|---|
| P/VP (e vs. mediana própria de 12 meses) | 0,85 → 100 · 1,00 → 70 · 1,15 → 0 | 15% | 20% | 15% |
| Dividend yield 12m | 6% → 0 · 12% → 100 | 5% | 15% | 30% |
| Regularidade, estabilidade (CV) e tendência dos rendimentos | composição 50/30/20 | 25% | 20% | 25% |
| Liquidez diária (R$) | 300 mil → 0 · 5 mi → 100 | 15% | 10% | 10% |
| Efeito na diversificação (fundo e segmento após a compra) | ≤ metade do limite → 100 · no limite → 0 | 20% | 15% | 5% |
| Vacância, inadimplência, alavancagem | média das disponíveis | 15% | 15% | 10% |
| Preço vs. média de 12 meses | −10% → 100 · +10% → 0 | 5% | 5% | 5% |

Fator sem dado não entra na nota: o peso é redistribuído e a **cobertura** é exibida. Com cobertura abaixo de 50%, o fundo não é recomendado.

**Vetos ("não comprar")**, que nenhuma nota compensa: corte de rendimentos (últimos 3 meses vs. anteriores ≤ −15%, −12% no perfil Renda); P/VP acima de 1,05 (Conservador) ou 1,10; liquidez abaixo de R$ 300 mil/dia (R$ 500 mil no Conservador); vacância, inadimplência ou alavancagem acima dos limites; fundo ou segmento acima do limite de concentração de Planejamento; sem cotação.

**Alocação:** percorre os elegíveis por nota e compra o máximo de cotas que mantém fundo e segmento dentro dos limites; a sobra passa ao próximo (até 3 sugestões). Cada sugestão traz cotas, valor, saldo, renda incremental estimada, peso após a compra, justificativa, riscos e a tabela de fatores com o dado e a fonte usados.

## 8. Fundos novos e Assistente

**Triagem** (`domain/screener.py`, `services/universe_service.py`):
- **Universo:** todos os FIIs, Fiagros e FI-Infra listados (BRAPI `/quote/list`, sem token; 385 fundos em 14/10/2026). O segmento, o patrimônio, os cotistas e o VP vêm do informe mensal da CVM, casado pelo ISIN, porque o setor da BRAPI não é confiável para FIIs.
- **Filtros padrão:** liquidez no pregão ≥ R$ 300 mil, patrimônio ≥ R$ 300 mi, ≥ 10 mil cotistas, P/VP entre 0,70 e 1,10, exclui o que já está na carteira. Cada reprovação tem motivo.
- **Avaliação:** os finalistas recebem cotação, 12 meses de histórico e proventos (BRAPI com token) e entram no mesmo motor de recomendação, como ativos inativos (não poluem a lista de acompanhamento). Sem dados suficientes, o motor recusa recomendar.

**Assistente** (`assistant/agent.py`, `assistant/tools.py`):
- **Funcionamento:** loop de chamadas de ferramentas na API Chat Completions da OpenAI. A chave vem de `OPENAI_API_KEY`; o modelo, de `INVESTAI_OPENAI_MODEL` (padrão `gpt-4.1-mini`, a confirmar na conta). Máximo de 6 rodadas por pergunta.
- **Ferramentas:** carteira, orçamento, proventos, recomendações, triagem de fundos novos, avaliação de candidatos, simulador, lista de acompanhamento e alertas. O modelo não tem SQL nem internet.
- **Regras do prompt:** nenhum número sem ferramenta; avisar quando há dados DEMO; fundos novos só se o motor os considerar elegíveis; fontes e limitações ao final. A interface mostra quais ferramentas cada resposta consultou.

## 9. Limitações conhecidas

- Sem autenticação (uso local, usuário único). JWT/Cognito entra antes de qualquer deploy público.
- Rentabilidade total é **retorno simples** sobre capital aplicado; TWR e comparação com CDI/IFIX/IPCA ficam para a Fase 2.
- Recibos de subscrição podem ter direito a rendimentos pro rata; o MVP não os considera elegíveis (conservador).
- Fiagros (ex.: RURA11) estão como classe FII para simplificar; podem ganhar classe própria.
- O extrato de Movimentação da B3 importa apenas rendimentos e amortizações; compras/vendas devem vir do extrato de Negociação (evita contagem dupla).
- Visões recalculam tudo a cada requisição — adequado para uma carteira pessoal; cache/materialização se o volume crescer.
