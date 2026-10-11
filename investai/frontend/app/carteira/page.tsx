"use client";

import Link from "next/link";
import { Lines } from "@/components/charts";
import { Badge, Card, Empty, ErrorBox, Loading, Notes, PageHeader, ShareBars, SourceBadge, Td, Th } from "@/components/ui";
import { useApi } from "@/lib/api";
import { date, money, monthLabel, pct, qty, signClass } from "@/lib/format";
import type { Allocation, Position } from "@/lib/types";

export default function CarteiraPage() {
  const { data, error, loading } = useApi<{ reference_date: string; positions: Position[]; warnings: string[] }>("/api/portfolio/positions");
  const alloc = useApi<Allocation>("/api/portfolio/allocation");
  const hist = useApi<{ month: string; invested: string; market_value: string | null; dividends_cum: string }[]>("/api/portfolio/history");
  if (error) return <ErrorBox error={error} />;
  if (loading || !data) return <Loading />;
  const rows = data.positions;

  return (
    <>
      <PageHeader title="Minha carteira" subtitle={`Posições calculadas a partir das movimentações · referência ${date(data.reference_date)}`} />
      <Card>
        {rows.length ? (
          <div className="-mx-5 overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <Th>Ativo</Th>
                  <Th right>Cotas</Th>
                  <Th right>Preço médio</Th>
                  <Th right>Custo</Th>
                  <Th right>Cotação</Th>
                  <Th right>Valor de mercado</Th>
                  <Th right>Resultado</Th>
                  <Th right>DY 12m</Th>
                  <Th right>YoC 12m</Th>
                  <Th right>Peso</Th>
                </tr>
              </thead>
              <tbody>
                {rows.map((p) => (
                  <tr key={p.ticker}>
                    <Td>
                      <Link href={`/carteira/${p.ticker}`} className="font-medium text-ink hover:text-brand-700">
                        {p.ticker}
                      </Link>
                      <div className="text-xs text-ink-3">{p.segment ?? "Sem segmento"}</div>
                    </Td>
                    <Td right>
                      {qty(p.quantity)}
                      {Number(p.pending_quantity) > 0 && (
                        <div>
                          <Badge color="amber" title="Aguardando liquidação/conversão">+{qty(p.pending_quantity)} pend.</Badge>
                        </div>
                      )}
                      {Number(p.planned_quantity) > 0 && (
                        <div>
                          <Badge color="violet" title="Ordem planejada, não executada">+{qty(p.planned_quantity)} plan.</Badge>
                        </div>
                      )}
                    </Td>
                    <Td right>{p.cost_known ? money(p.avg_price) : <Badge color="amber">não informado</Badge>}</Td>
                    <Td right>{money(p.total_cost)}</Td>
                    <Td right>
                      {money(p.price)}
                      <div className="text-[11px] text-ink-3">
                        {p.price_date ? date(p.price_date) : ""} <SourceBadge source={p.price_source} />
                      </div>
                    </Td>
                    <Td right>{money(p.market_value)}</Td>
                    <Td right className={signClass(p.unrealized_pnl)}>
                      {money(p.unrealized_pnl)}
                      <div className="text-[11px]">{pct(p.unrealized_pct)}</div>
                    </Td>
                    <Td right>{pct(p.dividend_stats.dy_12m)}</Td>
                    <Td right>{pct(p.dividend_stats.yoc_12m)}</Td>
                    <Td right>{pct(p.weight, 1)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>
            Nenhuma posição. <Link href="/movimentacoes" className="text-brand-700 underline">Registre uma movimentação</Link>.
          </Empty>
        )}
      </Card>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Distribuição por ativo" right={alloc.data && <span className="text-xs text-ink-3">base: {alloc.data.basis.toLowerCase()}</span>}>
          {alloc.data?.by_asset.length ? <ShareBars items={alloc.data.by_asset} /> : <Empty>Sem dados.</Empty>}
          {alloc.data && (
            <>
              <h3 className="mb-2 mt-5 text-xs font-semibold uppercase text-ink-3">Por classe</h3>
              <ShareBars items={alloc.data.by_class} />
            </>
          )}
        </Card>
        <Card title="Evolução patrimonial" className="lg:col-span-2">
          {hist.data?.length ? (
            <Lines
              data={hist.data.map((h) => ({ label: monthLabel(h.month), Investido: Number(h.invested), Mercado: h.market_value === null ? null : Number(h.market_value) }))}
              series={[
                { key: "Investido", label: "Capital investido (custo)" },
                { key: "Mercado", label: "Valor de mercado" },
              ]}
            />
          ) : (
            <Empty>Sem histórico.</Empty>
          )}
          <Notes items={["Valor de mercado só aparece nos meses em que todos os ativos têm cotação registrada até o fim do mês. Custo desconhecido conta como zero no capital investido."]} />
        </Card>
      </div>
    </>
  );
}
