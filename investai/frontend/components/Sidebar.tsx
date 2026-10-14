"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const NAV: { href: string; label: string; icon: string; phase?: string }[] = [
  { href: "/", label: "Dashboard", icon: "◧" },
  { href: "/carteira", label: "Minha carteira", icon: "▤" },
  { href: "/dividendos", label: "Dividendos", icon: "◷" },
  { href: "/movimentacoes", label: "Movimentações", icon: "⇅" },
  { href: "/planejamento", label: "Planejamento", icon: "◎" },
  { href: "/recomendacoes", label: "Recomendações", icon: "✦" },
  { href: "/simulador", label: "Simulador", icon: "↗" },
  { href: "/assistente", label: "Assistente InvestAI", icon: "◌" },
  { href: "/configuracoes", label: "Configurações", icon: "⚙" },
];

export default function Sidebar() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));
  return (
    <>
      <div className="sticky top-0 z-30 flex items-center justify-between bg-navy-900 px-4 py-3 text-white lg:hidden">
        <Logo />
        <button aria-label="Abrir menu" onClick={() => setOpen(!open)} className="rounded-md px-2 py-1 ring-1 ring-white/20">
          ☰
        </button>
      </div>
      <aside
        className={`${open ? "block" : "hidden"} fixed inset-y-0 left-0 z-40 w-64 bg-navy-900 text-slate-200 lg:sticky lg:top-0 lg:block lg:h-screen`}
      >
        <div className="hidden px-5 py-5 lg:block">
          <Logo />
        </div>
        <nav className="mt-2 space-y-0.5 px-3">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              onClick={() => setOpen(false)}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                active(n.href) ? "bg-white/10 font-medium text-white" : "text-slate-300 hover:bg-white/5 hover:text-white"
              }`}
            >
              <span aria-hidden className="w-4 text-center text-brand-500">{n.icon}</span>
              <span className="flex-1">{n.label}</span>
              {n.phase && <span className="rounded bg-white/10 px-1.5 text-[10px] text-slate-300">{n.phase}</span>}
            </Link>
          ))}
        </nav>
        <p className="absolute bottom-4 left-0 right-0 px-5 text-[11px] leading-snug text-slate-400">
          Ferramenta de acompanhamento e apoio à decisão. Não executa ordens e não constitui recomendação de investimento.
        </p>
      </aside>
      {open && <div className="fixed inset-0 z-30 bg-black/40 lg:hidden" onClick={() => setOpen(false)} />}
    </>
  );
}

function Logo() {
  return (
    <div className="flex items-center gap-2">
      <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-500 text-sm font-bold text-white">AI</span>
      <span className="text-lg font-semibold tracking-tight text-white">
        Invest<span className="text-brand-500">AI</span>
      </span>
    </div>
  );
}
