"use client";

import { useState } from "react";
import { SimpleBars } from "@/components/charts";
import { Card, Empty, ErrorBox, Field, Loading, Notes, PageHeader, SourceBadge, Stat, Td, Th, btnCls, inputCls } from "@/components/ui";
import { api, useApi } from "@/lib/api";
import { date, money, pct, qty, signClass } from "@/lib/format";
import type { Position } from "@/lib/types";

interface Detail {
  asset: { ticker: string; name: string | null; segment: string | null; asset_class: string; manager: string | null; cnpj: string | null; notes: string | null };
  position: Position | null;
  dividend_stats: {
    last: { amount_per_share: string; com_date: string; ex_date: string | null; payment_date: string; source: string } | null;
    history_24m: { payment_date: string; amount_per_share: string }[];
    sum_12m: string;
    mean_monthly_12m: string | null;
    stdev_12m: string;
    coefficient_of_variation: string | null;
    regularity_12m: string;
    trend_last3_vs_prior: string | null;
    dy_monthly_last: string | null;
    dy_12m: string | null;
    dy_last_annualized: string | null;
    yoc_12m: string | null;
    data_points: number;
  };
  quotes: { quote_date: string; close: string; source: string }[];
  indicators: { available: boolean; fields: string[]; reason: string };
}

const SEGMENTS = ["Recebíveis (CRI)", "Logística", "Shoppings", "Híbrido", "Lajes corporativas", "Agro (Fiagro)", "Energia", "Fundo de fundos", "Renda urbana", "Outros"];

export default function FundoPage({ params }: { params: { ticker: string } }) {
  const t = params.ticker.toUpperCase();
  const { data, error, loading, reload } = useApi<Detail>(`/api/assets/${t}/detail`);
  const [price, setPrice] = useState("");
  const [qdate, setQdate] = useState(new Date().toISOString().slice(0, 10));
  const [msg, setMsg] = useState<string | null>(null);

  if (error) return <ErrorBox error={error} />;
  if (loading || !data) return <Loading />;
  const p = data.position;
  const s = data.dividend_stats;

  async function saveQuote(e: React.FormEvent) {
    e.preventDefault();
    try {
      await api("/api/quotes", { method: "POST", json: { ticker: t, quote_date: qdate, close: price.replace(",", ".") } });
      setPrice("");
      setMsg("Cotação registrada.");
      reload();
    } catch (err) {
      setMsg((err as Error).message);
    }
  }
  async function saveSegment(segment: string) {
    await api(`/api/assets/${t}`, { method: "PATCH", json: { segment } });
    reload();
  }

  return (
    <>
      <PageHeader
        title={`${t}${data.asset.name ? ` · ${data.asset.name}` : ""}`}
        subtitle={
          <span className="flex flex-wrap items-center gap-2">
            Gestora: {data.asset.manager ?? "—"} · Segmento:
            <select aria-label="Segmento" className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-sm" value={data.asset.segment ?? ""} onChange={(e) => saveSegment(e.target.value)}>
              <option value="">Não classificado</option>
              {[...new Set([...(data.asset.segment ? [data.asset.segment] : []), ...SEGMENTS])].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </span>
        }
      />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Cotas confirmadas" value={qty(p?.quantity)} hint={p && Number(p.pending_quantity) > 0 ? `+${qty(p.pending_quantity)} pendentes` : undefined} />
        <Stat label="Preço médio" value={p?.cost_known ? money(p.avg_price) : "não informado"} hint={`Custo: ${money(p?.total_cost)}`} />
        <Stat label="Valor de mercado" value={money(p?.market_value)} hint={p?.price ? <>Cotação {money(p.price)} em {date(p.price_date)} <SourceBadge source={p.price_source} /></> : "Sem cotação"} />
        <Stat label="Resultado não realizado" value={money(p?.unrealized_pnl)} tone={signClass(p?.unrealized_pnl)} hint={pct(p?.unrealized_pct)} />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Proventos por cota (24 meses)" className="lg:col-span-2">
          {s.history_24m.length ? (
            <SimpleBars data={s.history_24m.map((h) => ({ label: date(h.payment_date).slice(3), valor: Number(h.amount_per_share) }))} dataKey="valor" label="R$ por cota" />
          ) : (
            <Empty>Sem histórico de proventos. Cadastre anúncios em Dividendos.</Empty>
          )}
        </Card>
        <Card title="Indicadores de distribuição">
          <dl className="space-y-1.5 text-sm">
            <KV k="Último provento" v={s.last ? <>{money(s.last.amount_per_share)} <SourceBadge source={s.last.source} /></> : "—"} />
            <KV k="Data-com / Data-ex" v={s.last ? `${date(s.last.com_date)} / ${date(s.last.ex_date)}` : "—"} />
            <KV k="Pagamento" v={s.last ? date(s.last.payment_date) : "—"} />
            <KV k="Soma 12 meses" v={money(s.sum_12m)} />
            <KV k="Média mensal 12m" v={money(s.mean_monthly_12m)} />
            <KV k="Regularidade 12m" v={pct(s.regularity_12m, 0)} />
            <KV k="Volatilidade (CV)" v={pct(s.coefficient_of_variation, 1)} />
            <KV k="Tendência (3m vs anteriores)" v={pct(s.trend_last3_vs_prior, 1)} />
            <KV k="DY último (mensal)" v={pct(s.dy_monthly_last)} />
            <KV k="DY 12m (s/ cotação)" v={pct(s.dy_12m)} />
            <KV k="Yield on Cost 12m" v={pct(s.yoc_12m)} />
          </dl>
          <Notes items={[`${s.data_points} registro(s) de proventos usados.`]} />
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Registrar cotação manual">
          <form onSubmit={saveQuote} className="space-y-3">
            <Field label="Data">
              <input type="date" className={inputCls} value={qdate} onChange={(e) => setQdate(e.target.value)} required />
            </Field>
            <Field label="Preço de fechamento (R$)">
              <input className={inputCls} inputMode="decimal" value={price} onChange={(e) => setPrice(e.target.value)} placeholder="0,00" required />
            </Field>
            <button className={btnCls}>Salvar cotação</button>
            {msg && <p className="text-xs text-ink-2">{msg}</p>}
          </form>
        </Card>
        <Card title="Histórico de cotações">
          {data.quotes.length ? (
            <div className="max-h-64 overflow-y-auto">
              <table>
                <thead>
                  <tr>
                    <Th>Data</Th>
                    <Th right>Fechamento</Th>
                    <Th>Fonte</Th>
                  </tr>
                </thead>
                <tbody>
                  {data.quotes.map((q) => (
                    <tr key={q.quote_date + q.source}>
                      <Td>{date(q.quote_date)}</Td>
                      <Td right>{money(q.close)}</Td>
                      <Td>
                        <SourceBadge source={q.source} />
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>Sem cotações.</Empty>
          )}
        </Card>
        <Card title="Indicadores fundamentalistas">
          <ul className="space-y-1 text-sm">
            {data.indicators.fields.map((f) => (
              <li key={f} className="flex justify-between">
                <span className="text-ink-2">{f}</span>
                <span className="text-ink-3">indisponível</span>
              </li>
            ))}
          </ul>
          <Notes items={[data.indicators.reason]} />
        </Card>
      </div>
    </>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-ink-2">{k}</dt>
      <dd className="text-right tabular-nums text-ink">{v}</dd>
    </div>
  );
}
