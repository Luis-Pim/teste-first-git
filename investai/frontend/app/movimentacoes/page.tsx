"use client";

import { useState } from "react";
import { Badge, Card, Empty, ErrorBox, Field, Loading, PageHeader, StatusBadge, Td, Th, btnCls, btnGhostCls, inputCls } from "@/components/ui";
import { API_URL, api, useApi } from "@/lib/api";
import { TXN_LABEL, date, money, qty } from "@/lib/format";
import type { Transaction } from "@/lib/types";

const EMPTY = {
  type: "BUY",
  status: "CONFIRMED",
  trade_date: new Date().toISOString().slice(0, 10),
  ticker: "",
  quantity: "",
  unit_price: "",
  fees: "0",
  total_amount: "",
  ratio_from: "",
  ratio_to: "",
  notes: "",
};
type Form = typeof EMPTY;

const NEEDS = {
  ticker: ["OPENING_BALANCE", "BUY", "SELL", "SUBSCRIPTION", "PREEMPTIVE_RIGHT", "SUBSCRIPTION_LEFTOVER", "AMORTIZATION", "SPLIT", "REVERSE_SPLIT", "DIVIDEND"],
  quantity: ["OPENING_BALANCE", "BUY", "SELL", "SUBSCRIPTION", "PREEMPTIVE_RIGHT", "SUBSCRIPTION_LEFTOVER", "DIVIDEND"],
  price: ["OPENING_BALANCE", "BUY", "SELL", "SUBSCRIPTION", "PREEMPTIVE_RIGHT", "SUBSCRIPTION_LEFTOVER"],
  ratio: ["SPLIT", "REVERSE_SPLIT"],
};

const dec = (v: string) => (v.trim() === "" ? null : v.replace(/\./g, "").replace(",", ".").trim());
// aceita "1.046,80" e "1046.80"
const decSmart = (v: string) => (v.includes(",") ? dec(v) : v.trim() === "" ? null : v.trim());

export default function MovimentacoesPage() {
  const [filter, setFilter] = useState("");
  const { data, error, loading, reload } = useApi<Transaction[]>(`/api/transactions${filter ? `?status=${filter}` : ""}`);
  const [form, setForm] = useState<Form>(EMPTY);
  const [editing, setEditing] = useState<number | null>(null);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  function edit(t: Transaction) {
    setEditing(t.id);
    setForm({
      type: t.type,
      status: t.status,
      trade_date: t.trade_date,
      ticker: t.ticker ?? "",
      quantity: t.quantity ?? "",
      unit_price: t.unit_price ?? "",
      fees: t.fees ?? "0",
      total_amount: t.total_amount ?? "",
      ratio_from: t.ratio_from ?? "",
      ratio_to: t.ratio_to ?? "",
      notes: t.notes ?? "",
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const body = {
      type: form.type,
      status: form.status,
      trade_date: form.trade_date,
      ticker: form.ticker.trim() ? form.ticker.toUpperCase() : null,
      quantity: decSmart(form.quantity),
      unit_price: decSmart(form.unit_price),
      fees: decSmart(form.fees) ?? "0",
      total_amount: decSmart(form.total_amount),
      ratio_from: decSmart(form.ratio_from),
      ratio_to: decSmart(form.ratio_to),
      notes: form.notes || null,
    };
    try {
      if (editing) await api(`/api/transactions/${editing}`, { method: "PUT", json: body });
      else await api("/api/transactions", { method: "POST", json: body });
      setMsg({ ok: true, text: editing ? "Operação atualizada." : "Operação registrada." });
      setEditing(null);
      setForm({ ...EMPTY, type: form.type, trade_date: form.trade_date });
      reload();
    } catch (err) {
      setMsg({ ok: false, text: (err as Error).message });
    }
  }

  async function del(id: number) {
    if (!confirm("Excluir esta operação? A posição será recalculada.")) return;
    try {
      await api(`/api/transactions/${id}`, { method: "DELETE" });
      reload();
    } catch (err) {
      setMsg({ ok: false, text: (err as Error).message });
    }
  }

  async function confirmTxn(t: Transaction) {
    const price = (t.unit_price ?? t.total_amount) ? null : prompt("Preço executado por cota (R$):");
    try {
      await api(`/api/transactions/${t.id}/confirm`, { method: "POST", json: price ? { unit_price: price.replace(",", ".") } : {} });
      reload();
    } catch (err) {
      setMsg({ ok: false, text: (err as Error).message });
    }
  }

  const show = (k: keyof typeof NEEDS) => NEEDS[k].includes(form.type);

  return (
    <>
      <PageHeader title="Movimentações" subtitle="Cada operação recalcula saldo de cotas, preço médio e histórico. Vendas acima do saldo são bloqueadas." />

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title={editing ? `Editar operação #${editing}` : "Nova operação"} className="lg:col-span-2">
          <form onSubmit={submit} className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Field label="Tipo">
              <select className={inputCls} value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })}>
                {Object.entries(TXN_LABEL).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Status">
              <select className={inputCls} value={form.status} onChange={(e) => setForm({ ...form, status: e.target.value })}>
                <option value="CONFIRMED">Confirmada</option>
                <option value="PENDING">Pendente (liquidação/conversão)</option>
                <option value="PLANNED">Planejada (não executada)</option>
              </select>
            </Field>
            <Field label="Data">
              <input type="date" className={inputCls} value={form.trade_date} onChange={(e) => setForm({ ...form, trade_date: e.target.value })} required />
            </Field>
            {show("ticker") && (
              <Field label="Ativo">
                <input className={inputCls} value={form.ticker} onChange={(e) => setForm({ ...form, ticker: e.target.value })} placeholder="KNRI11" />
              </Field>
            )}
            {show("quantity") && (
              <Field label="Quantidade">
                <input className={inputCls} inputMode="decimal" value={form.quantity} onChange={(e) => setForm({ ...form, quantity: e.target.value })} />
              </Field>
            )}
            {show("price") && (
              <Field label="Preço unitário (R$)" hint={form.type === "OPENING_BALANCE" ? "Seu preço médio atual" : undefined}>
                <input className={inputCls} inputMode="decimal" value={form.unit_price} onChange={(e) => setForm({ ...form, unit_price: e.target.value })} />
              </Field>
            )}
            {show("price") && (
              <Field label="Taxas (R$)">
                <input className={inputCls} inputMode="decimal" value={form.fees} onChange={(e) => setForm({ ...form, fees: e.target.value })} />
              </Field>
            )}
            {!show("ratio") && (
              <Field label="Valor total (R$)" hint={show("price") ? "Opcional: substitui qtd × preço + taxas" : undefined}>
                <input className={inputCls} inputMode="decimal" value={form.total_amount} onChange={(e) => setForm({ ...form, total_amount: e.target.value })} />
              </Field>
            )}
            {show("ratio") && (
              <>
                <Field label="Proporção de">
                  <input className={inputCls} value={form.ratio_from} onChange={(e) => setForm({ ...form, ratio_from: e.target.value })} placeholder="1" />
                </Field>
                <Field label="Para">
                  <input className={inputCls} value={form.ratio_to} onChange={(e) => setForm({ ...form, ratio_to: e.target.value })} placeholder="10" />
                </Field>
              </>
            )}
            <div className="col-span-2 md:col-span-4">
              <Field label="Observação">
                <input className={inputCls} value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
              </Field>
            </div>
            <div className="col-span-2 flex items-center gap-2 md:col-span-4">
              <button className={btnCls}>{editing ? "Salvar alterações" : "Registrar"}</button>
              {editing && (
                <button type="button" className={btnGhostCls} onClick={() => (setEditing(null), setForm(EMPTY))}>
                  Cancelar
                </button>
              )}
              {msg && <span className={`text-sm ${msg.ok ? "text-emerald-700" : "text-red-700"}`}>{msg.text}</span>}
            </div>
          </form>
        </Card>
        <ImportCard onDone={reload} />
      </div>

      <Card
        className="mt-4"
        title="Histórico"
        right={
          <select aria-label="Filtrar status" className="rounded-md border border-slate-300 px-2 py-1 text-sm" value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="">Todas</option>
            <option value="CONFIRMED">Confirmadas</option>
            <option value="PENDING">Pendentes</option>
            <option value="PLANNED">Planejadas</option>
          </select>
        }
      >
        {error && <ErrorBox error={error} />}
        {loading && <Loading />}
        {data && !data.length && <Empty>Nenhuma operação.</Empty>}
        {data && data.length > 0 && (
          <div className="-mx-5 overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <Th>Data</Th>
                  <Th>Tipo</Th>
                  <Th>Status</Th>
                  <Th>Ativo</Th>
                  <Th right>Qtd</Th>
                  <Th right>Preço</Th>
                  <Th right>Valor</Th>
                  <Th>Origem</Th>
                  <Th />
                </tr>
              </thead>
              <tbody>
                {data.map((t) => (
                  <tr key={t.id}>
                    <Td>{date(t.trade_date)}</Td>
                    <Td>{TXN_LABEL[t.type] ?? t.type}</Td>
                    <Td>
                      <StatusBadge status={t.status} />
                    </Td>
                    <Td className="font-medium">{t.ticker ?? "—"}</Td>
                    <Td right>{t.ratio_from ? `${qty(t.ratio_from)}:${qty(t.ratio_to)}` : qty(t.quantity)}</Td>
                    <Td right>{t.unit_price ? money(t.unit_price) : t.type === "OPENING_BALANCE" ? <Badge color="amber">sem PM</Badge> : "—"}</Td>
                    <Td right>{money(t.total_amount)}</Td>
                    <Td>
                      <span className="text-xs text-ink-3" title={t.notes ?? ""}>
                        {t.source}
                      </span>
                    </Td>
                    <Td right>
                      <div className="flex justify-end gap-2 text-xs">
                        {t.status !== "CONFIRMED" && (
                          <button className="font-medium text-brand-700 hover:underline" onClick={() => confirmTxn(t)}>
                            confirmar
                          </button>
                        )}
                        <button className="text-ink-2 hover:underline" onClick={() => edit(t)}>
                          editar
                        </button>
                        <button className="text-red-700 hover:underline" onClick={() => del(t.id)}>
                          excluir
                        </button>
                      </div>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}

interface Preview {
  format: string;
  counts: Record<string, number>;
  rows: { row: number; status: string; data: Record<string, string | null>; errors: string[]; skipped_reason: string | null }[];
  reconciliation: { ticker: string; before: string; after: string; delta: string }[];
  consistency_error: string | null;
}

const ROW_COLOR: Record<string, "green" | "amber" | "red" | "gray" | "violet"> = { NOVA: "green", DUPLICADA: "gray", POSSIVEL_DUPLICATA: "amber", ERRO: "red", IGNORADA: "violet" };

function ImportCard({ onDone }: { onDone: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [incDup, setIncDup] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function send(path: string) {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    if (path.endsWith("commit")) fd.append("include_possible_duplicates", String(incDup));
    return api<Preview & { imported: number; duplicates_skipped: number }>(path, { method: "POST", body: fd });
  }

  return (
    <Card title="Importar CSV / Excel">
      <p className="mb-3 text-xs text-ink-2">
        Aceita o <a className="text-brand-700 underline" href={`${API_URL}/api/imports/template`}>modelo InvestAI</a> e os extratos de Negociação e Movimentação da Área do Investidor B3. Duplicidades são detectadas antes de gravar.
      </p>
      <input type="file" accept=".csv,.xlsx,.xls" className="block w-full text-sm" onChange={(e) => (setFile(e.target.files?.[0] ?? null), setPreview(null), setMsg(null))} />
      <div className="mt-3 flex gap-2">
        <button
          className={btnGhostCls}
          disabled={!file}
          onClick={async () => {
            try {
              setPreview((await send("/api/imports/preview")) ?? null);
            } catch (e) {
              setMsg((e as Error).message);
            }
          }}
        >
          Pré-visualizar
        </button>
        <button
          className={btnCls}
          disabled={!preview || !!preview.consistency_error}
          onClick={async () => {
            try {
              const r = await send("/api/imports/commit");
              setMsg(`${r?.imported} operação(ões) importada(s); ${r?.duplicates_skipped} duplicada(s) ignorada(s).`);
              setPreview(null);
              onDone();
            } catch (e) {
              setMsg((e as Error).message);
            }
          }}
        >
          Importar
        </button>
      </div>
      {msg && <p className="mt-2 text-sm text-ink-2">{msg}</p>}
      {preview && (
        <div className="mt-3 space-y-2 text-sm">
          <div className="flex flex-wrap gap-1">
            <Badge color="blue">{preview.format}</Badge>
            {Object.entries(preview.counts).map(([k, v]) => (
              <Badge key={k} color={ROW_COLOR[k]}>
                {k.replace("_", " ").toLowerCase()}: {v}
              </Badge>
            ))}
          </div>
          {preview.consistency_error && <p className="text-red-700">{preview.consistency_error}</p>}
          {preview.reconciliation.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-ink-3">Conciliação (cotas)</div>
              {preview.reconciliation.map((r) => (
                <div key={r.ticker} className="text-xs">
                  {r.ticker}: {qty(r.before)} → {qty(r.after)} ({Number(r.delta) > 0 ? "+" : ""}
                  {qty(r.delta)})
                </div>
              ))}
            </div>
          )}
          {preview.counts.POSSIVEL_DUPLICATA && (
            <label className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={incDup} onChange={(e) => setIncDup(e.target.checked)} /> Importar também possíveis duplicatas de lançamentos manuais
            </label>
          )}
          <ul className="max-h-48 overflow-y-auto text-xs">
            {preview.rows.filter((r) => r.status !== "NOVA").map((r) => (
              <li key={r.row}>
                Linha {r.row}: <Badge color={ROW_COLOR[r.status]}>{r.status}</Badge> {r.errors.join("; ")} {r.skipped_reason}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
