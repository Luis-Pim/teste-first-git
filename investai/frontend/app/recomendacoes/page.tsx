import { Card, PageHeader } from "@/components/ui";

const CRITERIA = [
  "Concentração atual por fundo e por segmento",
  "Qualidade dos ativos, vacância e inadimplência",
  "Endividamento e sustentabilidade dos dividendos",
  "Histórico, regularidade e tendência das distribuições",
  "P/VP, dividend yield e preço frente ao histórico",
  "Liquidez e riscos específicos de cada fundo",
  "Impacto da compra na diversificação",
];

export default function RecomendacoesPage() {
  return (
    <>
      <PageHeader title="Recomendações de aporte" subtitle="Fase 3 — ainda não ativo." />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Por que ainda não está ativo">
          <p className="text-sm text-ink-2">
            Recomendações precisam de indicadores verificáveis (P/VP, vacância, inadimplência, alavancagem, liquidez) que só entram com as integrações CVM e
            relatórios gerenciais (Fases 2 e 3). Exibir uma recomendação sem esses dados seria inventar fundamento — por isso esta tela fica desligada até lá.
          </p>
          <p className="mt-3 text-sm text-ink-2">
            Hoje você já tem, em <strong>Planejamento</strong>, o orçamento disponível e quantas cotas de cada ativo cabem nele (aritmética simples, sem juízo de valor).
          </p>
        </Card>
        <Card title="Como o motor vai funcionar">
          <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
            {CRITERIA.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
          <p className="mt-3 text-sm text-ink-2">
            Perfis Conservador, Equilibrado e Renda ponderam esses fatores de forma diferente. Cada sugestão mostrará cotas, valor, saldo restante, renda incremental
            estimada, nova distribuição, justificativa e riscos — e o motor poderá indicar <em>não comprar</em> quando os indicadores forem desfavoráveis. Nunca
            ranqueia apenas pelo maior dividend yield.
          </p>
        </Card>
      </div>
    </>
  );
}
