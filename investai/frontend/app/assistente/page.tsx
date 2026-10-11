import { Card, PageHeader } from "@/components/ui";

const QUESTIONS = [
  "Quanto minha carteira está rendendo?",
  "Quais fundos estão concentrando meu patrimônio?",
  "Onde seria mais interessante aportar R$ 500 neste mês?",
  "Quais fundos reduziram seus dividendos?",
  "O que acontece se eu aumentar meu aporte para R$ 600?",
  "Quanto falta para atingir R$ 1.000 mensais?",
];

export default function AssistentePage() {
  return (
    <>
      <PageHeader title="Assistente InvestAI" subtitle="Fase 3 — arquitetura preparada, ainda não conectado a um modelo de linguagem." />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Arquitetura planejada">
          <ol className="list-decimal space-y-1.5 pl-5 text-sm text-ink-2">
            <li>O modelo recebe a pergunta e só pode responder chamando ferramentas — nunca de memória.</li>
            <li>Ferramentas = os mesmos endpoints desta API (posições, resumo, proventos, orçamento, simulador), com parâmetros validados; nada de SQL livre.</li>
            <li>RAG sobre relatórios gerenciais e informes CVM para vacância, inadimplência e riscos.</li>
            <li>Toda resposta traz data de referência, fontes usadas e limitações; valores DEMO são sinalizados.</li>
            <li>Provedor configurável por variável de ambiente; chaves no Secrets Manager.</li>
          </ol>
        </Card>
        <Card title="Perguntas que ele vai responder">
          <ul className="space-y-1.5 text-sm">
            {QUESTIONS.map((q) => (
              <li key={q} className="rounded-lg bg-slate-50 px-3 py-2 text-ink-2">“{q}”</li>
            ))}
          </ul>
        </Card>
      </div>
    </>
  );
}
