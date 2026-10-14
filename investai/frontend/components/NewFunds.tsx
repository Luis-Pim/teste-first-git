"use client";

import { useState } from "react";
import { Badge, Card, Empty, ErrorBox, Field, Notes, Td, Th, btnCls, btnGhostCls, inputCls } from "@/components/ui";
import { api } from "@/lib/api";
import { money, pct } from "@/lib/format";

interface Candidate {
  ticker: string;
  name: string | null;
  sub_type: string;
  segment: string | null;
  close: string | null;
  listed_liquidity: string;
  net_worth: string | null;
  shareholders: number | null;
  pvp: string | null;
  notes: string[];
}
interface Screen {
  universe_size: number;
  passed: number;
  shortlist: Candidate[];
  rejected_by_first_reason: Record<string, number>;
  listed_at: string | null;
  notes: string[];
}
interface Eval {
  ticker: string;
  score: string | null;
  coverage: string;
  eligible: boolean;
  vetoes: string[];
  risks: string[];
}
interface EvalResult {
  candidates: string[];
  picks: { ticker: string; quantity: number; cost: string; justification: string; incremental_monthly_income: string | null }[];
  ranking: Eval[];
  notes: string[];
  fetch: { quotes_saved: number; dividends_saved: number; missing: Record<string, string> };
}

const mi = (v: string | null) => (v === null ? "—" : `R$ ${(Number(v) / 1_000_000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} mi`);

export default function NewFunds({ profile }: { profile: string }) {
  const [segment, setSegment] = useState("");
  const [fiagro, setFiagro] = useState(false);
  const [minLiq, setMinLiq] = useState("500000");
  const [screen, setScreen] = useState<Screen | null>(null);
  const [sel, setSel] = useState<string[]>([]);
  const [evalRes, setEvalRes] = useState<EvalResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [watched, setWatched] = useState<string[]>([]);

  async function run(fn: () => Promise<void>, label: string) {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const doScreen = () =>
    run(async () => {
      const qs = new URLSearchParams({ min_liquidity: minLiq || "0", include_fiagro: String(fiagro), max_results: "15" });
      if (segment.trim()) qs.set("segment", segment.trim());
      const r = await api<Screen>(`/api/universe/screen?${qs}`);
      setScreen(r);
      setSel(r.shortlist.slice(0, 5).map((c) => c.ticker));
      setEvalRes(null);
    }, "screen");

  const doRefresh = () =>
    run(async () => {
      const r = await api<{ listed: number; with_cvm: number; warnings: string[] }>("/api/universe/refresh", { method: "POST" });
      if (r.warnings.length) setError(r.warnings.join(" "));
      await doScreen();
    }, "refresh");

  const doEval = () =>
    run(async () => {
      setEvalRes(await api<EvalResult>("/api/universe/evaluate", { method: "POST", json: { tickers: sel, profile } }));
    }, "eval");

  const watch = (t: string) =>
    run(async () => {
      await api(`/api/watchlist/${t}`, { method: "POST" });
      setWatched([...watched, t]);
    }, "watch");

  const evalByTicker = Object.fromEntries((evalRes?.ranking ?? []).map((e) => [e.ticker, e]));

  return (
    <Card title="Fundos novos" className="mt-4" right={<span className="text-xs text-ink-3">FIIs que você ainda não tem</span>}>
      <form
        className="flex flex-wrap items-end gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          doScreen();
        }}
      >
        <Field label="Segmento (opcional)" hint="Ex.: logíst, shopping, lajes, receb">
          <input id="nf-seg" className={inputCls} value={segment} onChange={(e) => setSegment(e.target.value)} />
        </Field>
        <Field label="Liquidez mínima no pregão (R$)">
          <input id="nf-liq" className={inputCls} inputMode="numeric" value={minLiq} onChange={(e) => setMinLiq(e.target.value.replace(/\D/g, ""))} />
        </Field>
        <label className="mb-6 flex items-center gap-2 text-sm text-ink-2">
          <input type="checkbox" checked={fiagro} onChange={(e) => setFiagro(e.target.checked)} /> Incluir Fiagros
        </label>
        <div className="mb-[18px] flex gap-2">
          <button className={btnCls} disabled={!!busy}>
            {busy === "screen" ? "Triando…" : "Fazer triagem"}
          </button>
          <button type="button" className={btnGhostCls} onClick={doRefresh} disabled={!!busy}>
            {busy === "refresh" ? "Atualizando…" : "Atualizar universo"}
          </button>
        </div>
      </form>
      {error && <ErrorBox error={error} />}

      {screen && (
        <>
          <p className="mt-2 text-sm text-ink-2">
            {screen.passed} de {screen.universe_size} fundos passaram nos filtros.{" "}
            {Object.entries(screen.rejected_by_first_reason)
              .map(([k, v]) => `${v} por ${k}`)
              .join(" · ")}
          </p>
          {screen.shortlist.length ? (
            <div className="-mx-5 mt-3 overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    <Th />
                    <Th>Fundo</Th>
                    <Th>Segmento (CVM)</Th>
                    <Th right>Cotação</Th>
                    <Th right>Liquidez no pregão</Th>
                    <Th right>Patrimônio</Th>
                    <Th right>Cotistas</Th>
                    <Th right>P/VP</Th>
                    <Th>Avaliação do motor</Th>
                    <Th />
                  </tr>
                </thead>
                <tbody>
                  {screen.shortlist.map((c) => {
                    const ev = evalByTicker[c.ticker];
                    return (
                      <tr key={c.ticker}>
                        <Td>
                          <input
                            type="checkbox"
                            aria-label={`Selecionar ${c.ticker}`}
                            checked={sel.includes(c.ticker)}
                            onChange={(e) => setSel(e.target.checked ? [...sel, c.ticker].slice(0, 10) : sel.filter((x) => x !== c.ticker))}
                          />
                        </Td>
                        <Td>
                          <span className="font-medium">{c.ticker}</span>
                          {c.sub_type !== "fii" && <Badge color="blue">{c.sub_type}</Badge>}
                          <div className="max-w-[14rem] truncate text-xs text-ink-3">{c.name}</div>
                        </Td>
                        <Td>{c.segment ?? <span className="text-ink-3">sem CVM</span>}</Td>
                        <Td right>{money(c.close)}</Td>
                        <Td right>{mi(c.listed_liquidity)}</Td>
                        <Td right>{mi(c.net_worth)}</Td>
                        <Td right>{c.shareholders?.toLocaleString("pt-BR") ?? "—"}</Td>
                        <Td right>{c.pvp ? Number(c.pvp).toFixed(2).replace(".", ",") : "—"}</Td>
                        <Td className="min-w-[14rem] whitespace-normal">
                          {!ev ? (
                            <span className="text-xs text-ink-3">não avaliado</span>
                          ) : ev.eligible ? (
                            <span className="text-xs">
                              <Badge color="green">elegível</Badge> nota {Number(ev.score).toFixed(0)} · cobertura {pct(ev.coverage, 0)}
                            </span>
                          ) : (
                            <span className="text-xs text-red-800">
                              <Badge color="red">não comprar</Badge> {ev.vetoes[0]}
                            </span>
                          )}
                        </Td>
                        <Td right>
                          {watched.includes(c.ticker) ? (
                            <Badge color="green">acompanhando</Badge>
                          ) : (
                            <button className="text-xs font-medium text-brand-700 hover:underline" onClick={() => watch(c.ticker)} disabled={!!busy}>
                              acompanhar
                            </button>
                          )}
                        </Td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>Nenhum fundo passou nos filtros. Afrouxe a liquidez ou o segmento, ou atualize o universo.</Empty>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <button className={btnCls} onClick={doEval} disabled={!!busy || sel.length === 0}>
              {busy === "eval" ? "Buscando dados e avaliando…" : `Avaliar ${sel.length} selecionado(s) no motor`}
            </button>
            <span className="text-xs text-ink-3">Busca cotação, 12 meses de histórico e proventos na BRAPI (consome cota do plano) e roda o motor no perfil atual.</span>
          </div>
          {evalRes && (
            <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm font-semibold text-ink">Resultado com os candidatos incluídos</div>
              {evalRes.picks.length ? (
                <ul className="mt-2 space-y-1 text-sm">
                  {evalRes.picks.map((p) => (
                    <li key={p.ticker}>
                      <b>{p.ticker}</b>: {p.quantity} cotas por {money(p.cost)}
                      {p.incremental_monthly_income && <> · +{money(p.incremental_monthly_income)}/mês estimados</>}
                      {evalRes.candidates.includes(p.ticker) && <Badge color="blue">novo</Badge>}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-ink-2">Nenhuma compra sugerida com o orçamento atual.</p>
              )}
              <Notes items={evalRes.notes} />
            </div>
          )}
          <Notes items={screen.notes} />
        </>
      )}
      {!screen && <Notes items={["A triagem parte de todos os FIIs listados na B3 (via BRAPI) e, quando disponível, do informe mensal da CVM."]} />}
    </Card>
  );
}
