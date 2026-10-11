"use client";

import { useState } from "react";
import { IncomeBars } from "@/components/charts";
import { Badge, Card, Empty, ErrorBox, Field, Loading, Notes, PageHeader, SourceBadge, Td, Th, btnCls, btnGhostCls, inputCls } from "@/components/ui";
import { api, useApi } from "@/lib/api";
import { date, money, monthLabel, qty } from "@/lib/format";
import type { ConfirmedIncome } from "@/lib/types";

interface Income {
  reference_date: string;
  received: { ticker: string; payment_date: string; amount: string; quantity: string | null; transaction_id: number }[];
  confirmed: ConfirmedIncome[];
  estimated: { ticker: string; month: string; amount_per_share: string; quantity: string; amount: string }[];
  calendar: { month: string; RECEBIDA: string; CONFIRMADA: string; ESTIMADA: string }[];
  notes: string[];
}
interface Ann {
  id: number;
  ticker: string;
  kind: string;
  amount_per_share: string;
  com_date: string;
  ex_date: string | null;
  payment_date: string;
  source: string;
}

export default function DividendosPage() {
  const inc = useApi<Income>("/api/dividends/income?horizon_months=12");
  const anns = useApi<Ann[]>("/api/dividends/announcements");
  const [form, setForm] = useState({ ticker: "", amount_per_share: "", com_date: "", ex_date: "", payment_date: "", kind: "RENDIMENTO" });
  const [msg, setMsg] = useState<string | null>(null);
  const reloadAll = () => {
    inc.reload();
    anns.reload();
  };

  if (inc.error) return <ErrorBox error={inc.error} />;
  if (inc.loading || !inc.data) return <Loading />;
  const d = inc.data;

  const estByMonth = Object.entries(
    d.estimated.reduce<Record<string, number>>((acc, e) => ({ ...acc, [e.month]: (acc[e.month] ?? 0) + Number(e.amount) }), {}),
  );

  async function addAnn(e: React.FormEvent) {
    e.preventDefault();
    try {
      await api("/api/dividends/announcements", {
        method: "POST",
        json: { ...form, amount_per_share: form.amount_per_share.replace(",", "."), ex_date: form.ex_date || null, ticker: form.ticker.toUpperCase() },
      });
      setMsg("Anúncio cadastrado.");
      setForm({ ...form, amount_per_share: "" });
      reloadAll();
    } catch (err) {
      setMsg((err as Error).message);
    }
  }
  async function markReceived(c: ConfirmedIncome) {
    await api("/api/transactions", {
      method: "POST",
      json: { type: "DIVIDEND", trade_date: c.payment_date, ticker: c.ticker, quantity: c.eligible_quantity, total_amount: c.amount, announcement_id: c.announcement_id, notes: "Registrado a partir do anúncio" },
    });
    reloadAll();
  }
  async function delAnn(id: number) {
    await api(`/api/dividends/announcements/${id}`, { method: "DELETE" });
    reloadAll();
  }

  return (
    <>
      <PageHeader title="Dividendos e calendário" subtitle="Três categorias que nunca se misturam: recebida (registrada), confirmada (anúncio oficial) e estimada (premissa histórica)." />

      <Card title="Calendário de recebimentos">
        {d.calendar.length ? <IncomeBars data={d.calendar} height={280} /> : <Empty>Sem dados ainda.</Empty>}
        <Notes items={d.notes} />
      </Card>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Confirmados (anunciados)">
          {d.confirmed.length ? (
            <div className="-mx-5 overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    <Th>Ativo</Th>
                    <Th>Data-com</Th>
                    <Th>Pagamento</Th>
                    <Th right>Cotas elegíveis</Th>
                    <Th right>Valor</Th>
                    <Th />
                  </tr>
                </thead>
                <tbody>
                  {d.confirmed.map((c) => (
                    <tr key={c.announcement_id}>
                      <Td>
                        <span className="font-medium">{c.ticker}</span> <SourceBadge source={c.source} />
                        {c.kind === "AMORTIZACAO" && <Badge color="blue">amortização</Badge>}
                      </Td>
                      <Td>{date(c.com_date)}</Td>
                      <Td>{date(c.payment_date)}</Td>
                      <Td right>
                        {qty(c.eligible_quantity)}
                        {c.provisional_quantity && <div className="text-[11px] text-amber-700">provisória (data-com futura)</div>}
                      </Td>
                      <Td right>
                        {money(c.amount)}
                        <div className="text-[11px] text-ink-3">{money(c.amount_per_share)}/cota</div>
                      </Td>
                      <Td right>
                        {c.status === "PAGO_NAO_REGISTRADO" ? (
                          <button className={btnGhostCls} onClick={() => markReceived(c)}>
                            Registrar recebido
                          </button>
                        ) : (
                          <Badge color="green">a receber</Badge>
                        )}
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>Nenhum anúncio com cotas elegíveis.</Empty>
          )}
        </Card>

        <Card title="Recebidos (registrados)">
          {d.received.length ? (
            <table>
              <thead>
                <tr>
                  <Th>Ativo</Th>
                  <Th>Pagamento</Th>
                  <Th right>Cotas</Th>
                  <Th right>Valor</Th>
                </tr>
              </thead>
              <tbody>
                {d.received.slice(0, 24).map((r) => (
                  <tr key={r.transaction_id}>
                    <Td>{r.ticker}</Td>
                    <Td>{date(r.payment_date)}</Td>
                    <Td right>{qty(r.quantity)}</Td>
                    <Td right>{money(r.amount)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty>Nenhum rendimento registrado. Use “Registrar recebido” ou lance em Movimentações.</Empty>
          )}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Renda estimada por mês">
          {estByMonth.length ? (
            <ul className="divide-y divide-slate-100 text-sm">
              {estByMonth.map(([m, v]) => (
                <li key={m} className="flex justify-between py-1.5">
                  <span className="text-ink-2">{monthLabel(m)}</span>
                  <span className="tabular-nums">{money(v)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <Empty>Sem histórico suficiente para estimar.</Empty>
          )}
        </Card>

        <Card title="Cadastrar anúncio de provento">
          <form onSubmit={addAnn} className="grid grid-cols-2 gap-3">
            <Field label="Ativo">
              <input className={inputCls} value={form.ticker} onChange={(e) => setForm({ ...form, ticker: e.target.value })} placeholder="KNRI11" required />
            </Field>
            <Field label="Tipo">
              <select className={inputCls} value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value })}>
                <option value="RENDIMENTO">Rendimento</option>
                <option value="AMORTIZACAO">Amortização</option>
              </select>
            </Field>
            <Field label="Valor por cota (R$)">
              <input className={inputCls} inputMode="decimal" value={form.amount_per_share} onChange={(e) => setForm({ ...form, amount_per_share: e.target.value })} required />
            </Field>
            <Field label="Data-com">
              <input type="date" className={inputCls} value={form.com_date} onChange={(e) => setForm({ ...form, com_date: e.target.value })} required />
            </Field>
            <Field label="Data-ex">
              <input type="date" className={inputCls} value={form.ex_date} onChange={(e) => setForm({ ...form, ex_date: e.target.value })} />
            </Field>
            <Field label="Pagamento">
              <input type="date" className={inputCls} value={form.payment_date} onChange={(e) => setForm({ ...form, payment_date: e.target.value })} required />
            </Field>
            <div className="col-span-2">
              <button className={btnCls}>Cadastrar</button>
              {msg && <span className="ml-3 text-xs text-ink-2">{msg}</span>}
            </div>
          </form>
        </Card>

        <Card title="Anúncios cadastrados">
          <div className="max-h-80 overflow-y-auto">
            <ul className="divide-y divide-slate-100 text-sm">
              {(anns.data ?? []).map((a) => (
                <li key={a.id} className="flex items-center justify-between py-1.5">
                  <span>
                    <span className="font-medium">{a.ticker}</span> {money(a.amount_per_share)} · pag. {date(a.payment_date)} <SourceBadge source={a.source} />
                  </span>
                  <button className="text-xs text-red-700 hover:underline" onClick={() => delAnn(a.id)} aria-label={`Excluir anúncio ${a.ticker}`}>
                    excluir
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </Card>
      </div>
    </>
  );
}
