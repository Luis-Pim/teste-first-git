# InvestAI

Plataforma de acompanhamento, análise e planejamento de investimentos pessoais no mercado brasileiro, começando por Fundos Imobiliários (FIIs).

> Ferramenta de acompanhamento e apoio à decisão. **Não executa ordens reais** e não constitui recomendação de investimento.

**Status: Fase 1 concluída; Fase 2 (motor de recomendação + dados de mercado) implementada, aguardando validação com token da BRAPI e acesso à CVM.** Veja [docs/ARQUITETURA.md](docs/ARQUITETURA.md) para arquitetura, modelo de dados, fontes, custos e roadmap.

## Stack

| Camada | Tecnologia |
|---|---|
| Backend | Python 3.12+ · FastAPI · SQLAlchemy 2 · Pydantic 2 · Alembic · pandas |
| Banco | PostgreSQL 16 |
| Frontend | Next.js 14 (App Router) · React 18 · TypeScript · Tailwind CSS · Recharts |
| Testes | pytest (domínio + API contra Postgres real) |

## Rodando localmente

### Opção A — Docker (mais simples)

```bash
docker compose up --build
docker compose exec api python -m app.seed               # sua carteira inicial
docker compose exec api python -m app.seed --demo-market # opcional: cotações/proventos DEMO
```

Abra http://localhost:3000 (API em http://localhost:8000/docs).

### Opção B — Manual

Pré-requisitos: Python 3.12+, Node 20+, PostgreSQL 16.

```bash
# 1. Banco
createuser -P investai          # senha: investai (apenas local)
createdb -O investai investai
createdb -O investai investai_test

# 2. API
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
alembic upgrade head
python -m app.seed               # carteira inicial (quantidades informadas, sem preço médio)
python -m app.seed --demo-market # opcional: dados DEMO para explorar a interface
uvicorn app.main:app --reload --port 8000

# 3. Frontend (outro terminal)
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

### Testes

```bash
cd backend && pytest -q          # 65 testes: preço médio, vendas parciais, subscrições, recomendação,
                                 # data-com, orçamento, simulador, importação/duplicidade, API
cd frontend && npm run typecheck && npm run build
```

## Primeiros passos com a sua carteira

1. **Movimentações → editar** cada "Saldo inicial" e informe seu **preço médio**. Sem ele, P/L, Yield on Cost e rentabilidade total ficam indisponíveis (de propósito: o sistema não estima custo).
2. Quando as 10 cotas de **KNCR11** (13ª emissão) forem convertidas, clique em **confirmar** na operação pendente.
3. Quando executar a compra de **MXRF11**, clique em **confirmar** na ordem planejada e informe o preço executado.
4. Registre cotações (tela do fundo) e anúncios de proventos (Dividendos) — ou configure a BRAPI em `backend/.env`.
5. Em **Planejamento**, defina os valores dos dois aportes extraordinários.
6. Para remover os dados DEMO: `python -m app.seed --clear-demo`.

## Estrutura

```
investai/
├── backend/
│   ├── app/
│   │   ├── domain/        # regras puras (Decimal, sem I/O): posições, proventos, orçamento, simulador, importação
│   │   ├── services/      # orquestração ORM ↔ domínio, visões da API, atualização de mercado
│   │   ├── market/        # adaptadores: Manual, BRAPI, BCB SGS
│   │   ├── routers/       # endpoints FastAPI
│   │   ├── models.py      # SQLAlchemy
│   │   ├── schemas.py     # Pydantic (validação de entrada)
│   │   └── seed.py        # carga da carteira inicial (+ DEMO opcional)
│   ├── alembic/           # migrações
│   └── tests/
├── frontend/
│   ├── app/               # páginas (dashboard, carteira, fundo, dividendos, movimentações, ...)
│   ├── components/        # UI e gráficos
│   └── lib/               # cliente da API, formatação pt-BR, tipos
├── docs/ARQUITETURA.md
└── docker-compose.yml
```
