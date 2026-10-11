"use client";

import { useEffect, useState } from "react";
import { Lines } from "@/components/charts";
import { Card, ErrorBox, Field, Notes, PageHeader, Stat, Td, Th, btnCls, inputCls } from "@/components/ui";
import { api } from "@/lib/api";
import { money, monthLabel, pct } from "@/lib/format";

type Row = Record<string, string>;
interface SimResult {
  assumptions: Record<string, string>;
  monthly: Row[];
  yearly: (Row & { year: number })[];
  final: Row;
  disclaimer: string;
}
const SCEN = ["conservador", "base", "otimista"] as const;
type Scen = (typeof SCEN)[number];

interface Inputs {
  initial_value: string;
  monthly_contribution: string;
  annual_contribution_increase: string;
  reinvest_pct: string;
  years: number;
  extra1: string;
  extra2: string;
}

export default function SimuladorPage() {
  const [inp, setInp] = useState<Inputs>({ initial_value: "", monthly_contribution: "300", annual_contribution_increase: "0", reinvest_pct: "1", years: 15, extra1: "0", extra2: "0" });
  const [assump, setAssump] = useState<Record<Scen, Record<string, string>> | null>(null);
  const [res, setRes] = useState<{ meta: Record<string, string>; scenarios: Record<Scen, SimResult> } | null>(null);
  const [view, setView] = useState<Scen>("base");
  const [real, setReal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [goals, setGoals] = useState<{ t500?: Row & { reached: boolean; years: number }; c2000?: Row; extras?: Row } | null>(null);

  useEffect(() => {
    api<Record<Scen, Record<string, string>>>("/api/simulator/scenarios").then(setAssump).catch((e) => setError(e.message));
  }, []);

  function body() {
    const extras = [
      { month: 6, amount: inp.extra1 || "0" },
      { month: 12, amount: inp.extra2 || "0" },
    ].filter((e) => Number(e.amount) > 0);
    return {
      initial_value: inp.initial_value.trim() === "" ? null : inp.initial_value.replace(",", "."),
      monthly_contribution: inp.monthly_contribution.replace(",", "."),
      annual_contribution_increase: String(Number(inp.annual_contribution_increase) / 100),
      reinvest_pct: inp.reinvest_pct,
      years: inp.years,
      extras,
      scenario: view,
      scenario_overrides: assump ?? undefined,
    };
  }

  async function run() {
    try {
      setError(null);
      const b = body();
      const r = await api<{ meta: Record<string, string>; scenarios: Record<Scen, SimResult> }>("/api/simulator/run", { method: "POST", json: b });
      setRes(r);
      const goalBody = { ...b, assumptions: assump?.[view] };
      const [t500, c2000, extras] = await Promise.all([
        api<Row & { reached: boolean; years: number }>("/api/simulator/goal-time", { method: "POST", json: { ...goalBody, target_monthly_income: "500" } }),
        api<Row>("/api/simulator/goal-contribution", { method: "POST", json: { ...goalBody, target_monthly_income: "2000" } }),
        api<Row>("/api/simulator/extras-impact", { method: "POST", json: { ...goalBody, compare_extras: [{ month: 6, amount: "5000" }, { month: 12, amount: "5000" }] } }),
      ]);
      setGoals({ t500, c2000, extras });
    } catch (e) {
      setError((e as Error).message);
    }
  }

  useEffect(() => {
    if (assump) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assump, view]);

  const sim = res?.scenarios[view];
  const k = (base: string) => (real ? `${base}_real` : base);
  const chart = sim?.monthly
    .filter((_, i) => i % 3 === 2)
    .map((r) => ({ label: monthLabel(r.month), Patrimonio: Number(r[k("patrimonio")]), Aportado: Number(r.total_aportado), Reinvestido: Number(r.dividendos_reinvestidos_acum) }));
  const income = sim?.yearly.map((r) => ({ label: `Ano ${r.year}`, Mensal: Number(r[k("renda_mensal")]) }));

  return (
    <>
      <PageHeader title="Simulador patrimonial" subtitle="Projeções hipotéticas. Todas as premissas ficam visíveis e editáveis; nenhum yield é tratado como garantido." />
      {error && <ErrorBox error={error} />}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Parâmetros">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Patrimônio inicial (R$)" hint="Vazio = carteira atual">
              <input className={inputCls} value={inp.initial_value} onChange={(e) => setInp({ ...inp, initial_value: e.target.value })} placeholder="carteira atual" />
            </Field>
            <Field label="Aporte mensal (R$)">
              <input className={inputCls} value={inp.monthly_contribution} onChange={(e) => setInp({ ...inp, monthly_contribution: e.target.value })} />
            </Field>
            <Field label="Aumento anual do aporte (%)">
              <input className={inputCls} value={inp.annual_contribution_increase} onChange={(e) => setInp({ ...inp, annual_contribution_increase: e.target.value })} />
            </Field>
            <Field label="Reinvestimento">
              <select className={inputCls} value={inp.reinvest_pct} onChange={(e) => setInp({ ...inp, reinvest_pct: e.target.value })}>
                <option value="1">100% dos dividendos</option>
                <option value="0.5">50%</option>
                <option value="0">Não reinvestir</option>
              </select>
            </Field>
            <Field label="Extra em junho (R$)">
              <input className={inputCls} value={inp.extra1} onChange={(e) => setInp({ ...inp, extra1: e.target.value })} />
            </Field>
            <Field label="Extra em dezembro (R$)">
              <input className={inputCls} value={inp.extra2} onChange={(e) => setInp({ ...inp, extra2: e.target.value })} />
            </Field>
            <div className="col-span-2">
              <Field label={`Horizonte: ${inp.years} anos`}>
                <input type="range" min={1} max={30} value={inp.years} onChange={(e) => setInp({ ...inp, years: Number(e.target.value) })} className="w-full accent-brand-600" />
              </Field>
            </div>
          </div>

          {assump && (
            <div className="mt-4">
              <div className="mb-1 text-xs font-medium text-ink-2">Premissas por cenário (% ao ano)</div>
              <table className="text-xs">
                <thead>
                  <tr>
                    <Th />
                    {SCEN.map((s) => (
                      <Th key={s} right>
                        {s}
                      </Th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["dividend_yield_annual", "Yield inicial"],
                    ["dividend_growth_annual", "Cresc. dividendos"],
                    ["price_growth_annual", "Valorização cotas"],
                    ["inflation_annual", "Inflação"],
                  ].map(([key, label]) => (
                    <tr key={key}>
                      <td className="py-1 pr-2 text-ink-2">{label}</td>
                      {SCEN.map((s) => (
                        <td key={s} className="py-1 pl-1">
                          <input
                            aria-label={`${label} ${s}`}
                            className="w-16 rounded border border-slate-300 px-1.5 py-1 text-right text-xs"
                            value={String(Number((Number(assump[s][key]) * 100).toFixed(4)))}
                            onChange={(e) => setAssump({ ...assump, [s]: { ...assump[s], [key]: String(Number(e.target.value.replace(",", ".")) / 100) } })}
                          />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <button className={`${btnCls} mt-4 w-full`} onClick={run}>
            Simular
          </button>
          {res?.meta.initial_value_source && <Notes items={[`Patrimônio inicial: ${res.meta.initial_value_source}.`, ...(res.meta.warning ? [res.meta.warning] : [])]} />}
        </Card>

        <div className="space-y-4 lg:col-span-2">
          <div className="flex flex-wrap items-center gap-2">
            {SCEN.map((s) => (
              <button
                key={s}
                onClick={() => setView(s)}
                className={`rounded-full px-3 py-1 text-sm capitalize ${view === s ? "bg-navy-900 text-white" : "bg-white text-ink ring-1 ring-slate-300"}`}
              >
                {s}
              </button>
            ))}
            <label className="ml-auto flex items-center gap-2 text-sm text-ink-2">
              <input type="checkbox" checked={real} onChange={(e) => setReal(e.target.checked)} /> Valores reais (descontada a inflação)
            </label>
          </div>

          {sim && (
            <>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="Patrimônio final" value={money(sim.final[k("patrimonio")])} />
                <Stat label="Total aportado" value={money(sim.final.total_aportado)} />
                <Stat label="Renda mensal final" value={money(sim.final[k("renda_mensal")])} />
                <Stat label="Renda anual final" value={money(sim.final.renda_anual)} hint="nominal" />
              </div>
              <Card title="Patrimônio, total aportado e dividendos reinvestidos">
                <Lines
                  data={chart ?? []}
                  series={[
                    { key: "Patrimonio", label: real ? "Patrimônio (real)" : "Patrimônio" },
                    { key: "Aportado", label: "Total aportado", dashed: true },
                    { key: "Reinvestido", label: "Dividendos reinvestidos (acum.)" },
                  ]}
                />
              </Card>
              <Card title="Renda passiva mensal ao fim de cada ano">
                <Lines data={income ?? []} series={[{ key: "Mensal", label: "Renda mensal" }]} height={220} />
              </Card>
            </>
          )}

          {goals && (
            <div className="grid gap-3 md:grid-cols-3">
              <Card title="Quando chego a R$ 500/mês?">
                <div className="text-2xl font-semibold text-ink">{goals.t500?.reached ? `${String(goals.t500.years).replace(".", ",")} anos` : "Não em 30 anos"}</div>
                <p className="text-xs text-ink-2">{goals.t500?.reached ? `por volta de ${goals.t500.month}` : "com estas premissas"}</p>
              </Card>
              <Card title={`Aporte para R$ 2.000/mês em ${inp.years} anos`}>
                <div className="text-2xl font-semibold text-ink">{money(goals.c2000?.monthly_contribution)}</div>
                <p className="text-xs text-ink-2">por mês, mantendo as demais premissas</p>
              </Card>
              <Card title="Impacto de 2 extras de R$ 5.000/ano">
                <div className="text-xl font-semibold text-ink">+{money(goals.extras?.delta_renda_mensal)}<span className="text-sm font-normal text-ink-2">/mês</span></div>
                <p className="text-xs text-ink-2">+{money(goals.extras?.delta_patrimonio)} de patrimônio ao fim do horizonte</p>
              </Card>
            </div>
          )}

          {sim && (
            <Card title="Tabela anual">
              <div className="-mx-5 max-h-80 overflow-auto">
                <table>
                  <thead>
                    <tr>
                      <Th>Ano</Th>
                      <Th right>Patrimônio</Th>
                      <Th right>Patrimônio real</Th>
                      <Th right>Aportado</Th>
                      <Th right>Reinvestido</Th>
                      <Th right>Renda mensal</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {sim.yearly.map((y) => (
                      <tr key={y.year}>
                        <Td>{y.year}</Td>
                        <Td right>{money(y.patrimonio)}</Td>
                        <Td right>{money(y.patrimonio_real)}</Td>
                        <Td right>{money(y.total_aportado)}</Td>
                        <Td right>{money(y.dividendos_reinvestidos_acum)}</Td>
                        <Td right>{money(y.renda_mensal)}</Td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Notes
                items={[
                  sim.disclaimer,
                  `Premissas do cenário ${view}: yield ${pct(sim.assumptions.dividend_yield_annual, 1)}, crescimento de dividendos ${pct(sim.assumptions.dividend_growth_annual, 1)}, valorização ${pct(
                    sim.assumptions.price_growth_annual,
                    1,
                  )}, inflação ${pct(sim.assumptions.inflation_annual, 1)} ao ano.`,
                ]}
              />
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
