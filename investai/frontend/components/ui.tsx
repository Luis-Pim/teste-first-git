import type { ReactNode } from "react";

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-ink-2">{subtitle}</p>}
      </div>
      {actions && <div className="flex gap-2">{actions}</div>}
    </div>
  );
}

export function Card({ title, children, className = "", right }: { title?: ReactNode; children: ReactNode; className?: string; right?: ReactNode }) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      {(title || right) && (
        <div className="mb-4 flex items-center justify-between gap-2">
          {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-ink-3">{label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${tone ?? "text-ink"}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-ink-2">{hint}</div>}
    </div>
  );
}

const BADGE: Record<string, string> = {
  gray: "bg-slate-100 text-slate-700",
  green: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  amber: "bg-amber-50 text-amber-800 ring-amber-200",
  red: "bg-red-50 text-red-800 ring-red-200",
  blue: "bg-blue-50 text-blue-800 ring-blue-200",
  violet: "bg-violet-50 text-violet-800 ring-violet-200",
};

export function Badge({ children, color = "gray", title }: { children: ReactNode; color?: keyof typeof BADGE; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center rounded-md px-1.5 py-0.5 text-[11px] font-medium ring-1 ring-inset ring-transparent ${BADGE[color]}`}>
      {children}
    </span>
  );
}

/** Origem do dado — dados demonstrativos são sempre destacados. */
export function SourceBadge({ source }: { source?: string | null }) {
  if (!source) return null;
  if (source === "DEMO")
    return (
      <Badge color="amber" title="Dado demonstrativo — não é dado real de mercado">
        DEMO
      </Badge>
    );
  return <Badge color="gray">{source}</Badge>;
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, [string, keyof typeof BADGE]> = {
    CONFIRMED: ["Confirmada", "green"],
    PENDING: ["Pendente", "amber"],
    PLANNED: ["Planejada", "violet"],
  };
  const [label, color] = map[status] ?? [status, "gray"];
  return <Badge color={color}>{label}</Badge>;
}

export function Notes({ items }: { items?: string[] }) {
  if (!items?.length) return null;
  return (
    <ul className="mt-3 space-y-1 text-xs text-ink-2">
      {items.map((t, i) => (
        <li key={i} className="flex gap-1.5">
          <span aria-hidden>ⓘ</span>
          <span>{t}</span>
        </li>
      ))}
    </ul>
  );
}

export function Loading() {
  return <div className="animate-pulse rounded-xl bg-white p-8 text-sm text-ink-3">Carregando…</div>;
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
      <strong>Não foi possível carregar.</strong> {error}
      <div className="mt-1 text-xs text-red-700">Verifique se a API está rodando em {process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}.</div>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-ink-2">{children}</div>;
}

export const inputCls =
  "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-ink shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100";
export const btnCls =
  "inline-flex items-center justify-center gap-1.5 rounded-lg bg-brand-600 px-3.5 py-2 text-sm font-medium text-white shadow-sm hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50";
export const btnGhostCls =
  "inline-flex items-center justify-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-ink hover:bg-slate-50 disabled:opacity-50";

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-ink-2">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-ink-3">{hint}</span>}
    </label>
  );
}

export function Th({ children, right }: { children?: ReactNode; right?: boolean }) {
  return <th className={`whitespace-nowrap px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-ink-3 ${right ? "text-right" : "text-left"}`}>{children}</th>;
}

export function Td({ children, right, className = "" }: { children?: ReactNode; right?: boolean; className?: string }) {
  const wrap = className.includes("whitespace-") ? "" : "whitespace-nowrap";
  return <td className={`${wrap} px-3 py-2.5 text-sm ${right ? "text-right tabular-nums" : ""} ${className}`}>{children}</td>;
}

/** Barra horizontal de participação (alocação). Um único tom; rótulos em tinta de texto. */
export function ShareBars({ items }: { items: { label: string; weight: string | number | null }[] }) {
  const max = Math.max(...items.map((i) => Number(i.weight ?? 0)), 0.0001);
  return (
    <ul className="space-y-2">
      {items.map((i) => {
        const w = Number(i.weight ?? 0);
        return (
          <li key={i.label} className="grid grid-cols-[7.5rem_1fr_3.5rem] items-center gap-2 text-sm" title={`${i.label}: ${(w * 100).toFixed(1)}%`}>
            <span className="truncate text-ink-2">{i.label}</span>
            <span className="h-2.5 rounded-full bg-slate-100">
              <span className="block h-2.5 rounded-full bg-series-1" style={{ width: `${(w / max) * 100}%` }} />
            </span>
            <span className="text-right tabular-nums text-ink">{(w * 100).toFixed(1)}%</span>
          </li>
        );
      })}
    </ul>
  );
}

const ALERT_STYLE: Record<string, [string, string]> = {
  warning: ["border-amber-200 bg-amber-50", "⚠"],
  success: ["border-emerald-200 bg-emerald-50", "✓"],
  info: ["border-slate-200 bg-slate-50", "ⓘ"],
};

export function AlertList({ alerts }: { alerts: { severity: string; title: string; message: string }[] }) {
  if (!alerts.length) return <Empty>Nenhum alerta no momento.</Empty>;
  return (
    <ul className="space-y-2">
      {alerts.map((a, i) => {
        const [cls, icon] = ALERT_STYLE[a.severity] ?? ALERT_STYLE.info;
        return (
          <li key={i} className={`rounded-lg border px-3 py-2 ${cls}`}>
            <div className="text-sm font-medium text-ink">
              <span aria-hidden className="mr-1.5">{icon}</span>
              {a.title}
            </div>
            <div className="text-xs text-ink-2">{a.message}</div>
          </li>
        );
      })}
    </ul>
  );
}
