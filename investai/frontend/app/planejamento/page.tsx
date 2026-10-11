"use client";

import { useEffect, useState } from "react";
import { Badge, Card, Empty, ErrorBox, Field, Loading, Notes, PageHeader, Stat, Td, Th, btnCls, btnGhostCls, inputCls } from "@/components/ui";
import { api, useApi } from "@/lib/api";
import { money, monthLabel, pct } from "@/lib/format";
import type { LedgerRow } from "@/lib/types";

interface Strategy {
  monthly_contribution: string;
  reinvest_pct: string;
  annual_increase_pct: string;
  extra_contributions: { month: number; amount: string; label: string }[];
  start_month: string;
  initial_cash: string;
  profile: string;
  concentration_limit_asset: string;
  concentration_limit_segment: string;
}
interface Ledger {
  rows: LedgerRow[];
  current: LedgerRow | null;
  affordable: { ticker: string; price: string; max_quantity: number; cost: string; remaining: string }[];
  notes: string[];
}
const MONTHS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"];

export default function PlanejamentoPage() {
  const strat = useApi<Strategy>("/api/planning/strategy");
  const led = useApi<Ledger>("/api/planning/ledger?months_ahead=3");
  const [s, setS] = useState<Strategy | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => {
    if (strat.data) setS(strat.data);
  }, [strat.data]);

  if (strat.error || led.error) return <ErrorBox error={(strat.error ?? led.error)!} />;
  if (!s || !led.data) return <Loading />;
  const cur = led.data.current;

  async function save(e: React.FormEvent) {
    e.preventDefault();
    try {
      await api("/api/planning/strategy", { method: "PUT", json: s });
      setMsg("Estratégia salva.");
      strat.reload();
      led.reload();
    } catch (err) {
      setMsg((err as Error).message);
    }
  }
  const upd = (k: keyof Strategy, v: unknown) => setS({ ...s, [k]: v } as Strategy);
  const pctIn = (v: string) => String(Number(v) * 100);
  const pctOut = (v: string) => String(Number(v.replace(",", ".")) / 100);

  return (
    <>
      <PageHeader title="Planejamento de aportes" subtitle="Orçamento mensal = aporte próprio + dividendos reinvestidos + saldo anterior − compras realizadas." />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label={`Disponível em ${cur ? monthLabel(cur.month) : "—"}`} value={money(cur?.available)} hint={cur ? `Aporte ${cur.own_source.toLowerCase()}` : undefined} />
        <Stat label="Já utilizado no mês" value={money(cur?.spent)} />
        <Stat label="Saldo para o próximo mês" value={money(cur?.carry_out)} tone={cur && Number(cur.carry_out) < 0 ? "text-red-700" : undefined} />
        <Stat label="Ordens planejadas" value={money(cur?.planned_orders)} hint="Comprometido, ainda não descontado" />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Estratégia" className="lg:col-span-1">
          <form onSubmit={save} className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Aporte mensal (R$)">
                <input className={inputCls} inputMode="decimal" value={s.monthly_contribution} onChange={(e) => upd("monthly_contribution", e.target.value.replace(",", "."))} />
              </Field>
              <Field label="Reinvestir dividendos (%)">
                <input className={inputCls} inputMode="decimal" value={pctIn(s.reinvest_pct)} onChange={(e) => upd("reinvest_pct", pctOut(e.target.value))} />
              </Field>
              <Field label="Aumento anual do aporte (%)">
                <input className={inputCls} inputMode="decimal" value={pctIn(s.annual_increase_pct)} onChange={(e) => upd("annual_increase_pct", pctOut(e.target.value))} />
              </Field>
              <Field label="Início da estratégia">
                <input type="month" className={inputCls} value={s.start_month.slice(0, 7)} onChange={(e) => upd("start_month", `${e.target.value}-01`)} />
              </Field>
              <Field label="Saldo inicial em caixa (R$)">
                <input className={inputCls} inputMode="decimal" value={s.initial_cash} onChange={(e) => upd("initial_cash", e.target.value.replace(",", "."))} />
              </Field>
              <Field label="Perfil">
                <select className={inputCls} value={s.profile} onChange={(e) => upd("profile", e.target.value)}>
                  <option value="CONSERVADOR">Conservador</option>
                  <option value="EQUILIBRADO">Equilibrado</option>
                  <option value="RENDA">Renda</option>
                </select>
              </Field>
              <Field label="Limite por ativo (%)">
                <input className={inputCls} inputMode="decimal" value={pctIn(s.concentration_limit_asset)} onChange={(e) => upd("concentration_limit_asset", pctOut(e.target.value))} />
              </Field>
              <Field label="Limite por segmento (%)">
                <input className={inputCls} inputMode="decimal" value={pctIn(s.concentration_limit_segment)} onChange={(e) => upd("concentration_limit_segment", pctOut(e.target.value))} />
              </Field>
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-ink-2">Aportes extraordinários</div>
              {s.extra_contributions.map((x, i) => (
                <div key={i} className="mb-2 grid grid-cols-[5.5rem_1fr_auto] gap-2">
                  <select
                    aria-label="Mês"
                    className={inputCls}
                    value={x.month}
                    onChange={(e) => upd("extra_contributions", s.extra_contributions.map((y, j) => (j === i ? { ...y, month: Number(e.target.value) } : y)))}
                  >
                    {MONTHS.map((m, k) => (
                      <option key={m} value={k + 1}>
                        {m}
                      </option>
                    ))}
                  </select>
                  <input
                    aria-label="Valor"
                    className={inputCls}
                    inputMode="decimal"
                    value={x.amount}
                    onChange={(e) => upd("extra_contributions", s.extra_contributions.map((y, j) => (j === i ? { ...y, amount: e.target.value.replace(",", ".") } : y)))}
                  />
                  <button type="button" className={btnGhostCls} onClick={() => upd("extra_contributions", s.extra_contributions.filter((_, j) => j !== i))} aria-label="Remover">
                    ✕
                  </button>
                </div>
              ))}
              <button type="button" className={btnGhostCls} onClick={() => upd("extra_contributions", [...s.extra_contributions, { month: 6, amount: "0", label: "" }])}>
                + Adicionar extra
              </button>
            </div>
            <button className={btnCls}>Salvar estratégia</button>
            {msg && <span className="ml-3 text-xs text-ink-2">{msg}</span>}
          </form>
        </Card>

        <Card title="Razão mensal do orçamento" className="lg:col-span-2">
          <div className="-mx-5 overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <Th>Mês</Th>
                  <Th right>Saldo anterior</Th>
                  <Th right>Aporte próprio</Th>
                  <Th right>Dividendos reinv.</Th>
                  <Th right>Disponível</Th>
                  <Th right>Utilizado</Th>
                  <Th right>Saldo final</Th>
                </tr>
              </thead>
              <tbody>
                {led.data.rows.map((r) => (
                  <tr key={r.month} className={r.is_current ? "bg-brand-50/60" : ""}>
                    <Td>
                      {monthLabel(r.month)} {r.is_current && <Badge color="green">atual</Badge>}
                      {r.is_future && <Badge color="gray">previsto</Badge>}
                    </Td>
                    <Td right>{money(r.carry_in)}</Td>
                    <Td right>
                      {money(r.own_contribution)}
                      <div className="text-[11px] text-ink-3">{r.own_source === "PLANEJADO" ? "planejado" : "registrado"}{Number(r.planned_extra) > 0 && ` · inclui extra ${money(r.planned_extra)}`}</div>
                    </Td>
                    <Td right>{money(r.dividends_reinvested)}</Td>
                    <Td right className="font-medium">{money(r.available)}</Td>
                    <Td right>{money(r.spent)}</Td>
                    <Td right className={Number(r.carry_out) < 0 ? "text-red-700" : ""}>{money(r.carry_out)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Notes items={led.data.notes} />
        </Card>
      </div>

      <Card className="mt-4" title={`Quanto cabe no orçamento de ${cur ? money(cur.available) : "—"}`}>
        {led.data.affordable.length ? (
          <div className="-mx-5 overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <Th>Ativo</Th>
                  <Th right>Cotação</Th>
                  <Th right>Cotas possíveis</Th>
                  <Th right>Valor</Th>
                  <Th right>Sobra</Th>
                </tr>
              </thead>
              <tbody>
                {led.data.affordable.map((a) => (
                  <tr key={a.ticker}>
                    <Td className="font-medium">{a.ticker}</Td>
                    <Td right>{money(a.price)}</Td>
                    <Td right>{a.max_quantity}</Td>
                    <Td right>{money(a.cost)}</Td>
                    <Td right>{money(a.remaining)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>Registre cotações para calcular quantas cotas cabem no orçamento.</Empty>
        )}
        <Notes items={[`Reinvestimento configurado: ${pct(s.reinvest_pct, 0)}. A priorização de compras (motor de recomendações) chega na Fase 3.`]} />
      </Card>
    </>
  );
}
