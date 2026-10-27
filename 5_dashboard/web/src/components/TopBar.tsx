import { motion } from "motion/react";
import type { ModelsInfo } from "../api/client";
import { usePresent } from "../state/present";
import { Mark } from "./Mark";
import { ModelChip } from "./ModelChip";

export type Tab = "model" | "data" | "inspect" | "batch";
export const TABS: { id: Tab; label: string }[] = [
  { id: "model", label: "Model" }, { id: "data", label: "Data" }, { id: "inspect", label: "Inspect" }, { id: "batch", label: "Batch" },
];

export function TopBar({ tab, onTab, models, modelsError, onModel }: {
  tab: Tab; onTab(t: Tab): void; models: ModelsInfo | null; modelsError: string | null; onModel(id: string): void;
}) {
  const present = usePresent();
  return (
    <header className="topbar">
      <div className="brand">
        <Mark />
        <span className="brand-name">Aircraft Visual Inspection</span>
      </div>
      <nav className="tabs" role="tablist" aria-label="Views">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className="tab" onClick={() => onTab(t.id)}>
            {t.label}
            {tab === t.id && <motion.span layoutId="tab-underline" className="tab-underline" transition={{ type: "spring", stiffness: 500, damping: 40 }} />}
          </button>
        ))}
      </nav>
      <div className="topbar-right">
        <ModelChip info={models} error={modelsError} onSelect={onModel} />
        <button className="present-btn" aria-pressed={present.on} onClick={present.toggle}>
          <span className="label">Presentation</span><span className="key" aria-hidden="true">P</span>
        </button>
      </div>
    </header>
  );
}
