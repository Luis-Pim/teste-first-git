"use client";

import Link from "next/link";
import { useState } from "react";
import { Badge, Card, Empty, ErrorBox, Field, Loading, Notes, PageHeader, Stat, Td, Th, btnCls, inputCls } from "@/components/ui";
import NewFunds from "@/components/NewFunds";
import { useApi } from "@/lib/api";
import { date, money, pct } from "@/lib/format";

interface Factor {
  factor: string;
  label: string;
  weight: string;
  score: string | null;
  detail: string;
  source: string | null;
}
interface Evaluation {
  ticker: string;
  segment: string | null;
  price: string | null;
  score: string | null;
  coverage: string;
  eligible: boolean;
  vetoes: string[];
  risks: string[];
  factors: Factor[];
}
interface Pick extends Evaluation {
  quantity: number;
  cost: string;
  remaining_after: string;
  incremental_monthly_income: string | null;
  weight_after: string;
  justification: string;
}
interface Result {
  reference_date: string;
  profile: string;
  budget: string;
  budget_source: string;
  allocated: string;
  remaining: string;
  picks: Pick[];
  ranking: Evaluation[];
  distribution: { ticker: string; weight_before: string; weight_after: string }[];
  thresholds: Record<string, string>;
  notes: string[];
  disclaimer: string;
}

const PROFILES = [
  ["CONSERVADOR", "Conservador", "Qualidade, diversificação e estabilidade de renda"],
  ["EQUILIBRADO", "Equilibrado", "Equilíbrio entre renda e valorização"],
  ["RENDA", "Renda", "Dividendos sustentáveis, sem ignorar riscos"],
] as const;

function ScoreBar({ score }: { score: string | null }) {
  if (score === null) return <span className="text-xs text-ink-3">não avaliado</span>;
  const v = Number(score);
  return (
    <span className="flex items-center gap-2">
      <span className="h-1.5 w-20 rounded-full bg-slate-100">
        <span className="block h-1.5 rounded-full bg-series-1" style={{ width: `${v}%` }} />
      </span>
      <span className="w-8 text-right text-xs tabular-nums text-ink">{v.toFixed(0)}</span>
    </span>
  );
}

function FactorTable({ factors }: { factors: Factor[] }) {
  return (
    <div className="-mx-5 overflow-x-auto">
      <table>
        <thead>
          <tr>
            <Th>Fator</Th>
            <Th right>Peso</Th>
            <Th>Nota</Th>
            <Th>Dado usado</Th>
            <Th>Fonte</Th>
          </tr>
        </thead>
        <tbody>
          {factors.map((f) => (
            <tr key={f.factor}>
              <Td>{f.label}</Td>
              <Td right>{pct(f.weight, 0)}</Td>
              <Td>
                <ScoreBar score={f.score} />
              </Td>
              <Td className="min-w-[14rem] whitespace-normal text-xs text-ink-2">{f.detail}</Td>
              <Td className="whitespace-normal text-xs text-ink-3">
                {f.source?.startsWith("DEMO") ? <Badge color="amber">DEMO</Badge> : f.source ?? "—"}
              </Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RecomendacoesPage() {
  const [profile, setProfile] = useState<string>("");
  const [budget, setBudget] = useState("");
  const [query, setQuery] = useState("");
  const { data, error, loading } = useApi<Result>(`/api/recommendations${query}`);
  const [open, setOpen] = useState<string | null>(null);

  function apply(p = profile, b = budget) {
    const qs = new URLSearchParams();
    if (p) qs.set("profile", p);
    if (b.trim()) qs.set("budget", b.replace(/\./g, "").replace(",", "."));
    setQuery(qs.toString() ? `?${qs}` : "");
  }

  if (error) return <ErrorBox error={error} />;
  if (loading || !data) return <Loading />;
  const current = profile || data.profile;
  const vetoed = data.ranking.filter((e) => !e.eligible);
  const eligible = data.ranking.filter((e) => e.eligible);

  return (
    <>
      <PageHeader
        title="Recomendações de aporte"
        subtitle={`Referência ${date(data.reference_date)} · ${data.disclaimer}`}
      />

      <Card>
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <div className="mb-1 text-xs font-medium text-ink-2">Perfil</div>
            <div className="flex flex-wrap gap-2" role="group" aria-label="Perfil">
              {PROFILES.map(([k, label, hint]) => (
                <button
                  key={k}
                  title={hint}
                  aria-pressed={current === k}
                  onClick={() => (setProfile(k), apply(k))}
                  className={`rounded-full px-3 py-1.5 text-sm ${current === k ? "bg-navy-900 text-white" : "bg-white text-ink ring-1 ring-slate-300 hover:bg-slate-50"}`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <form
            className="flex items-end gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              apply();
            }}
          >
            <Field label="Valor a investir (R$)" hint={`Vazio = ${data.budget_source}`}>
              <input id="rec-budget" className={inputCls} inputMode="decimal" value={budget} onChange={(e) => setBudget(e.target.value)} placeholder={money(data.budget)} />
            </Field>
            <button className={`${btnCls} mb-[18px]`}>Recalcular</button>
          </form>
        </div>
        <Notes items={data.notes} />
      </Card>

      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Orçamento" value={money(data.budget)} hint={data.budget_source} />
        <Stat label="Sugerido para compra" value={money(data.allocated)} />
        <Stat label="Saldo restante" value={money(data.remaining)} hint="Fica para o próximo mês" />
        <Stat label="Fundos elegíveis" value={`${eligible.length} de ${data.ranking.length}`} hint={`${vetoed.length} com veto de compra`} />
      </div>

      <h2 className="mb-3 mt-6 text-sm font-semibold text-ink">Sugestões</h2>
      {data.picks.length ? (
        <div className={`grid gap-4 ${data.picks.length > 1 ? "xl:grid-cols-2" : ""}`}>
          {data.picks.map((p, i) => (
            <Card
              key={p.ticker}
              title={
                <span className="flex items-center gap-2">
                  <span className="grid h-6 w-6 place-items-center rounded-full bg-brand-600 text-xs text-white">{i + 1}</span>
                  <Link href={`/carteira/${p.ticker}`} className="hover:text-brand-700">
                    {p.ticker}
                  </Link>
                  <span className="text-xs font-normal text-ink-3">{p.segment}</span>
                </span>
              }
              right={<span className="text-xs text-ink-2">nota {Number(p.score).toFixed(0)} · cobertura {pct(p.coverage, 0)}</span>}
            >
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-sm sm:grid-cols-3">
                <div>
                  <dt className="text-xs text-ink-3">Comprar</dt>
                  <dd className="font-semibold tabular-nums">{p.quantity} cotas</dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-3">Valor</dt>
                  <dd className="tabular-nums">{money(p.cost)}</dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-3">Saldo depois</dt>
                  <dd className="tabular-nums">{money(p.remaining_after)}</dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-3">Renda extra estimada</dt>
                  <dd className="tabular-nums">{p.incremental_monthly_income ? `${money(p.incremental_monthly_income)}/mês` : "—"}</dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-3">Peso após a compra</dt>
                  <dd className="tabular-nums">{pct(p.weight_after, 1)}</dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-3">Cotação</dt>
                  <dd className="tabular-nums">{money(p.price)}</dd>
                </div>
              </dl>
              <p className="mt-3 text-sm text-ink-2">{p.justification}</p>
              {p.risks.length > 0 && (
                <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
                  <div className="text-xs font-semibold text-amber-900">Principais riscos</div>
                  <ul className="mt-1 list-disc pl-4 text-xs text-amber-900">
                    {p.risks.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
              <button className="mt-3 text-xs font-medium text-brand-700 hover:underline" onClick={() => setOpen(open === p.ticker ? null : p.ticker)}>
                {open === p.ticker ? "Ocultar" : "Ver"} os fatores avaliados
              </button>
              {open === p.ticker && (
                <div className="mt-3">
                  <FactorTable factors={p.factors} />
                </div>
              )}
            </Card>
          ))}
        </div>
      ) : (
        <Empty>
          Nenhuma compra sugerida com {money(data.budget)}: o orçamento não cobre uma cota de nenhum fundo elegível, ou todos receberam veto. Veja o ranking abaixo.
        </Empty>
      )}

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Ranking completo" className="lg:col-span-2">
          <div className="-mx-5 overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <Th>Fundo</Th>
                  <Th>Nota</Th>
                  <Th right>Cobertura</Th>
                  <Th>Situação</Th>
                </tr>
              </thead>
              <tbody>
                {data.ranking.map((e) => (
                  <tr key={e.ticker} className="cursor-pointer" onClick={() => setOpen(open === `r-${e.ticker}` ? null : `r-${e.ticker}`)}>
                    <Td>
                      <span className="font-medium">{e.ticker}</span>
                      <div className="text-xs text-ink-3">{e.segment ?? "Sem segmento"}</div>
                      {open === `r-${e.ticker}` && (
                        <div className="mt-2 max-w-2xl whitespace-normal" onClick={(ev) => ev.stopPropagation()}>
                          <FactorTable factors={e.factors} />
                        </div>
                      )}
                    </Td>
                    <Td>
                      <ScoreBar score={e.score} />
                    </Td>
                    <Td right>{pct(e.coverage, 0)}</Td>
                    <Td className="min-w-[16rem] whitespace-normal">
                      {e.eligible ? (
                        <Badge color="green">elegível</Badge>
                      ) : (
                        <div className="space-y-1">
                          <Badge color="red">não comprar</Badge>
                          {e.vetoes.map((v) => (
                            <div key={v} className="text-xs text-red-800">
                              {v}
                            </div>
                          ))}
                        </div>
                      )}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Notes items={["Clique em um fundo para ver nota, peso, dado e fonte de cada fator.", "Fatores sem dado não entram na nota; o peso deles é redistribuído e a cobertura cai."]} />
        </Card>

        <Card title="Distribuição antes e depois">
          <ul className="space-y-2 text-sm">
            {data.distribution.map((d) => {
              const changed = Math.abs(Number(d.weight_after) - Number(d.weight_before)) > 0.0001;
              return (
                <li key={d.ticker} className="grid grid-cols-[4.5rem_1fr] items-center gap-2">
                  <span className={changed ? "font-semibold text-ink" : "text-ink-2"}>{d.ticker}</span>
                  <span className="tabular-nums text-ink-2">
                    {pct(d.weight_before, 1)} → <span className={changed ? "font-semibold text-ink" : ""}>{pct(d.weight_after, 1)}</span>
                  </span>
                </li>
              );
            })}
          </ul>
          <h3 className="mb-2 mt-5 text-xs font-semibold uppercase text-ink-3">Limites do perfil</h3>
          <dl className="space-y-1 text-xs">
            <div className="flex justify-between">
              <dt className="text-ink-2">P/VP máximo</dt>
              <dd className="tabular-nums">{Number(data.thresholds.max_pvp).toFixed(2).replace(".", ",")}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-ink-2">Corte de rendimento (veto)</dt>
              <dd className="tabular-nums">{pct(data.thresholds.dividend_cut, 0)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-ink-2">Liquidez mínima/dia</dt>
              <dd className="tabular-nums">{money(data.thresholds.min_liquidity)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-ink-2">Vacância / inadimplência / alavancagem</dt>
              <dd className="tabular-nums">
                {pct(data.thresholds.max_vacancy, 0)} / {pct(data.thresholds.max_default, 0)} / {pct(data.thresholds.max_leverage, 0)}
              </dd>
            </div>
          </dl>
          <Notes items={["Limites de concentração por fundo e por segmento vêm de Planejamento."]} />
        </Card>
      </div>
      <NewFunds profile={current} />
    </>
  );
}
