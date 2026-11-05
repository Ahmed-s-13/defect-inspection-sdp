import { motion } from "motion/react";

export type Tab = "model" | "uncertainty" | "data" | "inspect" | "batch" | "helmet";
export const TABS: { id: Tab; label: string }[] = [
  { id: "model", label: "Model" }, { id: "uncertainty", label: "Uncertainty" }, { id: "data", label: "Data" }, { id: "inspect", label: "Inspect" }, { id: "batch", label: "Batch" },
  { id: "helmet", label: "Helmet" },
];

/** Static dash's top bar: the SAQR logo and name, and the tabs. One model, no presentation mode. */
export function TopBar({ tab, onTab }: { tab: Tab; onTab(t: Tab): void }) {
  return (
    <header className="topbar">
      <div className="brand">
        <img className="brand-mark" src="brand/saqr_mark.png" width={105} height={96} alt="" />
        <span className="brand-name">SAQR <span>AVI</span></span>
      </div>
      <nav className="tabs" role="tablist" aria-label="Views">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className="tab" onClick={() => onTab(t.id)}>
            {t.label}
            {tab === t.id && <motion.span layoutId="tab-underline" className="tab-underline" transition={{ type: "spring", stiffness: 500, damping: 40 }} />}
          </button>
        ))}
      </nav>
      <div className="topbar-right" aria-hidden="true" />
    </header>
  );
}
