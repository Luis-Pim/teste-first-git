"use client";

import { useEffect, useRef, useState } from "react";
import { Badge, Card, PageHeader, btnCls, inputCls } from "@/components/ui";
import { api, useApi } from "@/lib/api";

interface ToolTrace {
  tool: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  uses_demo_data: boolean;
}
interface Reply {
  answer: string;
  tools: ToolTrace[];
  model: string;
  reference_date: string;
  usage: { prompt_tokens: number; completion_tokens: number };
}
interface Msg {
  role: "user" | "assistant";
  content: string;
  meta?: Reply;
}

const TOOL_LABEL: Record<string, string> = {
  get_portfolio: "carteira",
  get_budget: "orçamento",
  get_dividends: "proventos",
  get_recommendations: "motor de recomendação",
  screen_new_funds: "triagem de fundos novos",
  evaluate_new_funds: "avaliação de candidatos",
  simulate: "simulador",
  add_to_watchlist: "lista de acompanhamento",
  get_alerts: "alertas",
};

const SUGGESTIONS = [
  "Onde seria mais interessante aportar R$ 500 neste mês?",
  "Quais fundos estão concentrando meu patrimônio?",
  "Me sugira um FII de logística que eu ainda não tenho.",
  "Algum fundo da carteira reduziu os dividendos?",
  "Em quanto tempo chego a R$ 1.000 por mês aportando R$ 600?",
  "Qual fundo tem o melhor equilíbrio entre risco e renda?",
];

const STORE_KEY = "investai-chat";

export default function AssistentePage() {
  const status = useApi<{ configured: boolean; model: string; tools: string[] }>("/api/assistant/status");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(STORE_KEY);
      if (saved) setMsgs(JSON.parse(saved));
    } catch {}
  }, []);
  useEffect(() => {
    try {
      sessionStorage.setItem(STORE_KEY, JSON.stringify(msgs.slice(-30)));
    } catch {}
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  async function send(text: string) {
    const q = text.trim();
    if (!q || busy) return;
    const next: Msg[] = [...msgs, { role: "user", content: q }];
    setMsgs(next);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const r = await api<Reply>("/api/assistant/chat", { method: "POST", json: { messages: next.map(({ role, content }) => ({ role, content })) } });
      setMsgs([...next, { role: "assistant", content: r.answer, meta: r }]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Assistente InvestAI"
        subtitle="Responde consultando os dados da sua carteira, o motor de recomendação e a triagem de fundos novos. Não inventa números e não executa ordens."
        actions={
          msgs.length > 0 && (
            <button className="text-sm text-ink-2 hover:underline" onClick={() => setMsgs([])}>
              Nova conversa
            </button>
          )
        }
      />
      {status.data && !status.data.configured && (
        <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          A chave da OpenAI não está configurada na API. Defina <code>OPENAI_API_KEY</code> no ambiente do backend e reinicie. O modelo usado será{" "}
          <code>{status.data.model}</code> (ajustável em <code>INVESTAI_OPENAI_MODEL</code>).
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <Card className="flex min-h-[28rem] flex-col">
          <div className="flex-1 space-y-4 overflow-y-auto pr-1" aria-live="polite">
            {msgs.length === 0 && (
              <div className="py-6 text-center text-sm text-ink-2">
                Pergunte sobre sua carteira, seus proventos, onde aportar ou peça sugestões de fundos novos.
              </div>
            )}
            {msgs.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-navy-900 px-4 py-2.5 text-sm text-white">
                  {m.content}
                </div>
              ) : (
                <div key={i} className="max-w-[92%]">
                  <div className="whitespace-pre-wrap rounded-2xl rounded-bl-sm border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-relaxed text-ink">{m.content}</div>
                  {m.meta && (
                    <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-3">
                      <span>Consultou:</span>
                      {m.meta.tools.length ? (
                        m.meta.tools.map((t, j) => (
                          <Badge key={j} color={!t.ok ? "red" : t.uses_demo_data ? "amber" : "gray"} title={JSON.stringify(t.arguments)}>
                            {TOOL_LABEL[t.tool] ?? t.tool}
                            {t.uses_demo_data ? " · DEMO" : ""}
                            {!t.ok ? " · erro" : ""}
                          </Badge>
                        ))
                      ) : (
                        <Badge color="amber">nenhuma ferramenta</Badge>
                      )}
                      <span>· {m.meta.model}</span>
                    </div>
                  )}
                </div>
              ),
            )}
            {busy && <div className="animate-pulse text-sm text-ink-3">Consultando seus dados…</div>}
            <div ref={end} />
          </div>
          {error && (
            <div role="alert" className="mt-3 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-800">
              {error}
            </div>
          )}
          <form
            className="mt-4 flex gap-2 border-t border-slate-100 pt-4"
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
          >
            <label htmlFor="chat-input" className="sr-only">
              Pergunta
            </label>
            <input id="chat-input" className={inputCls} value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ex.: onde aportar R$ 500 este mês?" disabled={busy} />
            <button className={btnCls} disabled={busy || !input.trim()}>
              Enviar
            </button>
          </form>
        </Card>

        <div className="space-y-4">
          <Card title="Experimente perguntar">
            <ul className="space-y-2">
              {SUGGESTIONS.map((s) => (
                <li key={s}>
                  <button className="w-full rounded-lg bg-slate-50 px-3 py-2 text-left text-sm text-ink-2 hover:bg-slate-100" onClick={() => send(s)} disabled={busy}>
                    {s}
                  </button>
                </li>
              ))}
            </ul>
          </Card>
          <Card title="Como ele responde">
            <ul className="list-disc space-y-1 pl-4 text-xs text-ink-2">
              <li>Só cita números que vieram das ferramentas; os selos abaixo de cada resposta mostram o que foi consultado.</li>
              <li>Fundos novos passam pela triagem e pelo mesmo motor de recomendação da carteira, com vetos de “não comprar”.</li>
              <li>Avisa quando os dados são DEMO e lista fontes e limitações no fim.</li>
            </ul>
          </Card>
        </div>
      </div>
    </>
  );
}
