// Valores chegam da API como string decimal; a conversão para number acontece só na exibição.
export type Num = string | number | null | undefined;

const brl = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL" });
const num = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 6 });

export const n = (v: Num): number | null => (v === null || v === undefined || v === "" ? null : Number(v));

export function money(v: Num, dash = "—"): string {
  const x = n(v);
  return x === null || Number.isNaN(x) ? dash : brl.format(x);
}

export function pct(v: Num, digits = 2, dash = "—"): string {
  const x = n(v);
  return x === null || Number.isNaN(x) ? dash : `${(x * 100).toLocaleString("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`;
}

export function qty(v: Num): string {
  const x = n(v);
  return x === null ? "—" : num.format(x);
}

export function date(v: string | null | undefined): string {
  if (!v) return "—";
  const [y, m, d] = v.slice(0, 10).split("-");
  return `${d}/${m}/${y}`;
}

export function monthLabel(key: string): string {
  const [y, m] = key.split("-");
  const names = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"];
  return `${names[Number(m) - 1]}/${y.slice(2)}`;
}

export const signClass = (v: Num) => {
  const x = n(v);
  if (x === null || x === 0) return "text-ink";
  return x > 0 ? "text-emerald-700" : "text-red-700";
};

export const TXN_LABEL: Record<string, string> = {
  OPENING_BALANCE: "Saldo inicial",
  BUY: "Compra",
  SELL: "Venda",
  SUBSCRIPTION: "Subscrição",
  PREEMPTIVE_RIGHT: "Direito de preferência",
  SUBSCRIPTION_LEFTOVER: "Sobras de subscrição",
  AMORTIZATION: "Amortização",
  SPLIT: "Desdobramento",
  REVERSE_SPLIT: "Grupamento",
  DIVIDEND: "Rendimento recebido",
  FEE: "Taxa / custo",
  CASH_DEPOSIT: "Aporte de dinheiro",
  CASH_WITHDRAWAL: "Retirada de dinheiro",
};

export const STATUS_LABEL: Record<string, string> = { CONFIRMED: "Confirmada", PENDING: "Pendente", PLANNED: "Planejada" };
