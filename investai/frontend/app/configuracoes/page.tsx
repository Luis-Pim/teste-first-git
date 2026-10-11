"use client";

import { useState } from "react";
import { Badge, Card, ErrorBox, Loading, Notes, PageHeader, Td, Th, btnCls } from "@/components/ui";
import { api, useApi } from "@/lib/api";

interface Sources {
  active_provider: string;
  brapi_token_configured: boolean;
  sources: { name: string; status: string; provides: string[]; notes: string }[];
}
interface Log {
  id: number;
  provider: string;
  operation: string;
  status: string;
  detail: string | null;
  started_at: string;
}

export default function ConfiguracoesPage() {
  const src = useApi<Sources>("/api/market/sources");
  const logs = useApi<Log[]>("/api/market/logs");
  const [result, setResult] = useState<string | null>(null);
  if (src.error) return <ErrorBox error={src.error} />;
  if (!src.data) return <Loading />;

  async function refresh(withDiv: boolean) {
    try {
      const r = await api<{ quotes_saved: number; dividends_saved: number; missing: Record<string, string> }>(`/api/market/refresh?with_dividends=${withDiv}`, { method: "POST" });
      const miss = Object.entries(r.missing);
      setResult(`${r.quotes_saved} cotação(ões), ${r.dividends_saved} provento(s) salvos.${miss.length ? ` Ausentes: ${miss.map(([t, m]) => `${t} (${m})`).join("; ")}` : ""}`);
      logs.reload();
    } catch (e) {
      setResult((e as Error).message);
    }
  }

  return (
    <>
      <PageHeader title="Configurações e integrações" subtitle={<>Provedor ativo: <Badge color="blue">{src.data.active_provider}</Badge> · Token BRAPI: {src.data.brapi_token_configured ? "configurado" : "não configurado"}</>} />
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Fontes de dados" className="lg:col-span-2">
          <div className="-mx-5 overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <Th>Fonte</Th>
                  <Th>Status</Th>
                  <Th>Fornece</Th>
                  <Th>Observações</Th>
                </tr>
              </thead>
              <tbody>
                {src.data.sources.map((s) => (
                  <tr key={s.name}>
                    <Td className="font-medium">{s.name}</Td>
                    <Td>
                      <Badge color={s.status === "ATIVO" ? "green" : s.status.startsWith("ADAPTADOR") ? "blue" : "gray"}>{s.status}</Badge>
                    </Td>
                    <Td className="whitespace-normal text-ink-2">{s.provides.join(", ")}</Td>
                    <Td className="whitespace-normal text-xs text-ink-2">{s.notes}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="Atualizar mercado">
          <p className="text-sm text-ink-2">Busca cotações dos ativos da carteira no provedor ativo, com cache e registro de execução.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button className={btnCls} onClick={() => refresh(false)}>Atualizar cotações</button>
            <button className={btnCls} onClick={() => refresh(true)}>Cotações + proventos</button>
          </div>
          {result && <p className="mt-3 text-sm text-ink-2">{result}</p>}
          <Notes items={["Para usar a BRAPI: defina INVESTAI_MARKET_PROVIDER=brapi e INVESTAI_BRAPI_TOKEN no .env da API (nunca no código)."]} />
        </Card>
      </div>
      <Card className="mt-4" title="Log de atualizações (rastreabilidade)">
        <div className="-mx-5 overflow-x-auto">
          <table>
            <thead>
              <tr>
                <Th>#</Th>
                <Th>Quando</Th>
                <Th>Provedor</Th>
                <Th>Status</Th>
                <Th>Detalhe</Th>
              </tr>
            </thead>
            <tbody>
              {(logs.data ?? []).map((l) => (
                <tr key={l.id}>
                  <Td>{l.id}</Td>
                  <Td>{new Date(l.started_at).toLocaleString("pt-BR")}</Td>
                  <Td>{l.provider}</Td>
                  <Td>
                    <Badge color={l.status === "OK" ? "green" : l.status === "PARTIAL" ? "amber" : "red"}>{l.status}</Badge>
                  </Td>
                  <Td className="max-w-xl truncate whitespace-nowrap text-xs text-ink-2">{l.detail}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
