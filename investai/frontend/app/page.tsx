"use client";

import Link from "next/link";
import { IncomeBars } from "@/components/charts";
import { AlertList, Card, Empty, ErrorBox, Loading, Notes, PageHeader, ShareBars, SourceBadge, Stat } from "@/components/ui";
import { useApi } from "@/lib/api";
import { date, money, monthLabel, pct, qty, signClass } from "@/lib/format";
import type { Alert, Allocation, ConfirmedIncome, LedgerRow, Summary } from "@/lib/types";

interface Dashboard {
  reference_date: string;
  summary: Summary;
  month_income: { received: string; confirmed: string; estimated: string };
  upcoming_payments: ConfirmedIncome[];
  budget: LedgerRow | null;
  allocation: Allocation;
  alerts: Alert[];
}
interface Income {
  calendar: { month: string; RECEBIDA: string; CONFIRMADA: string; ESTIMADA: string }[];
  notes: string[];
}

export default function DashboardPage() {
  const { data, error, loading } = useApi<Dashboard>("/api/dashboard");
  const inc = useApi<Income>("/api/dividends/income?horizon_months=6");
  if (error) return <ErrorBox error={error} />;
  if (loading || !data) return <Loading />;
  const s = data.summary;
  const demo = s.price_sources.includes("DEMO");

  return (
    <>
      <PageHeader
        title="Visão geral"
        subtitle={<>Data de referência {date(data.reference_date)} · Fontes de preço: {s.price_sources.length ? s.price_sources.map((p) => <SourceBadge key={p} source={p} />) : "nenhuma"}</>}
      />
      {demo && (
        <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-900">
          Cotações e proventos marcados como <strong>DEMO</strong> são ilustrativos e não representam o mercado. Remova-os com <code>python -m app.seed --clear-demo</code>.
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          label="Valor de mercado"
          value={money(s.market_value)}
          hint={s.market_value_complete ? `${s.positions_count} ativos com cotação` : `Parcial — sem cotação: ${s.missing_prices.join(", ")}`}
        />
        <Stat
          label="Capital investido"
          value={money(s.invested_total)}
          hint={
            s.cost_unknown.length ? (
              <span className="text-amber-700">Custo desconhecido em {s.cost_unknown.length} ativo(s) — informe o preço médio</span>
            ) : (
              <>Inclui {money(s.invested_pending)} pendente(s)</>
            )
          }
        />
        <Stat
          label="Resultado acumulado"
          value={s.total_return ? money(s.total_return.absolute) : "—"}
          tone={s.total_return ? signClass(s.total_return.absolute) : undefined}
          hint={s.total_return ? `${pct(s.total_return.pct)} incluindo proventos` : "Requer preço médio e cotação de todos os ativos"}
        />
        <Stat
          label="Renda passiva do mês"
          value={money(Number(data.month_income.received) + Number(data.month_income.confirmed))}
          hint={`${money(data.month_income.received)} recebida · ${money(data.month_income.confirmed)} confirmada`}
        />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Aporte disponível" right={data.budget && <span className="text-xs text-ink-3">{monthLabel(data.budget.month)}</span>}>
          {data.budget ? (
            <>
              <div className="text-3xl font-semibold tabular-nums text-ink">{money(data.budget.available)}</div>
              <dl className="mt-3 space-y-1 text-sm">
                <Row k={`Aporte próprio (${data.budget.own_source.toLowerCase()})`} v={money(data.budget.own_contribution)} />
                <Row k="Dividendos reinvestidos" v={money(data.budget.dividends_reinvested)} />
                <Row k="Saldo do mês anterior" v={money(data.budget.carry_in)} />
                <Row k="Já utilizado" v={money(data.budget.spent)} />
                <Row k="Saldo restante" v={<strong>{money(data.budget.carry_out)}</strong>} />
              </dl>
              <Link href="/planejamento" className="mt-3 inline-block text-sm font-medium text-brand-700 hover:underline">
                Ver planejamento →
              </Link>
            </>
          ) : (
            <Empty>Configure sua estratégia em Planejamento.</Empty>
          )}
        </Card>

        <Card title="Próximos pagamentos (confirmados)">
          {data.upcoming_payments.length ? (
            <ul className="divide-y divide-slate-100">
              {data.upcoming_payments.map((u) => (
                <li key={u.announcement_id} className="flex items-center justify-between py-2 text-sm">
                  <div>
                    <div className="font-medium text-ink">
                      {u.ticker} <SourceBadge source={u.source} />
                    </div>
                    <div className="text-xs text-ink-3">
                      {date(u.payment_date)} · {qty(u.eligible_quantity)} cotas × {money(u.amount_per_share)}
                      {u.provisional_quantity && " · qtd provisória"}
                    </div>
                  </div>
                  <span className="tabular-nums text-ink">{money(u.amount)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <Empty>Nenhum provento anunciado a receber. Cadastre anúncios em Dividendos.</Empty>
          )}
        </Card>

        <Card title="Alertas">
          <AlertList alerts={data.alerts} />
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Evolução e previsão de proventos" className="lg:col-span-2">
          {inc.data && inc.data.calendar.length ? <IncomeBars data={inc.data.calendar.slice(-12)} /> : <Empty>Sem proventos registrados ou anunciados.</Empty>}
          <Notes items={inc.data?.notes.slice(0, 1)} />
        </Card>
        <Card title="Distribuição por segmento" right={<span className="text-xs text-ink-3">base: {data.allocation.basis.toLowerCase()}</span>}>
          {data.allocation.by_segment.length ? <ShareBars items={data.allocation.by_segment} /> : <Empty>Sem posições.</Empty>}
          <Notes items={data.allocation.notes} />
        </Card>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Proventos no ano" value={money(s.dividends_ytd)} hint={`12 meses: ${money(s.dividends_12m)}`} />
        <Stat label="DY 12 meses" value={pct(s.dy_12m)} hint="Recebidos ÷ valor de mercado" />
        <Stat label="Yield on Cost 12m" value={pct(s.yoc_12m)} hint="Recebidos ÷ custo de aquisição" />
        <Stat label="vs. CDI / IFIX / IPCA" value="—" hint={s.benchmarks.reason} />
      </div>
      <Notes items={s.notes} />
    </>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-ink-2">{k}</dt>
      <dd className="tabular-nums text-ink">{v}</dd>
    </div>
  );
}
