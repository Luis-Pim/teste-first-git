"use client";

import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { money, monthLabel } from "@/lib/format";

// Paleta categórica validada (slots 1–3 passam CVD em todos os pares). Texto nunca usa a cor da série.
export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];
const AXIS = { fontSize: 11, fill: "#52606d" };
const GRID = "#e8ebef";
const inkLegend = (v: string) => <span style={{ color: "#52606d" }}>{v}</span>;

const compact = (v: number) =>
  Math.abs(v) >= 1000 ? `R$ ${(v / 1000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} mil` : `R$ ${v.toLocaleString("pt-BR", { maximumFractionDigits: 0 })}`;

type Row = Record<string, string | number | null>;

/** Renda por mês em três categorias que nunca se misturam: recebida, confirmada, estimada. */
export function IncomeBars({ data, height = 260 }: { data: { month: string; RECEBIDA: string; CONFIRMADA: string; ESTIMADA: string }[]; height?: number }) {
  const rows = data.map((d) => ({ month: monthLabel(d.month), Recebida: Number(d.RECEBIDA), Confirmada: Number(d.CONFIRMADA), Estimada: Number(d.ESTIMADA) }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }} barCategoryGap="30%">
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="month" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} tickFormatter={compact} width={72} />
        <Tooltip formatter={(v: number) => money(v)} cursor={{ fill: "rgba(15,27,45,0.04)" }} />
        <Legend iconType="square" formatter={inkLegend} wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="Recebida" stackId="a" fill={SERIES[0]} stroke="#fff" strokeWidth={1} />
        <Bar dataKey="Confirmada" stackId="a" fill={SERIES[1]} stroke="#fff" strokeWidth={1} />
        <Bar dataKey="Estimada" stackId="a" fill={SERIES[2]} fillOpacity={0.55} stroke="#fff" strokeWidth={1} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SimpleBars({ data, dataKey, label, height = 220 }: { data: Row[]; dataKey: string; label: string; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="label" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} tickFormatter={compact} width={72} />
        <Tooltip formatter={(v: number) => money(v)} cursor={{ fill: "rgba(15,27,45,0.04)" }} />
        <Bar dataKey={dataKey} name={label} fill={SERIES[0]} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Lines({
  data,
  series,
  height = 300,
  xKey = "label",
}: {
  data: Row[];
  series: { key: string; label: string; dashed?: boolean }[];
  height?: number;
  xKey?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey={xKey} tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} minTickGap={24} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} tickFormatter={compact} width={80} />
        <Tooltip formatter={(v: number) => money(v)} />
        {series.length > 1 && <Legend iconType="plainline" formatter={inkLegend} wrapperStyle={{ fontSize: 12 }} />}
        {series.map((s, i) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={SERIES[i % SERIES.length]}
            strokeWidth={2}
            strokeDasharray={s.dashed ? "5 4" : undefined}
            dot={data.length < 4 ? { r: 4 } : false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: "#fff" }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
