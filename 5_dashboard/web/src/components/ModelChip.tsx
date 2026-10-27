import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import type { ModelsInfo } from "../api/client";

export function ModelChip({ info, error, onSelect }: { info: ModelsInfo | null; error: string | null; onSelect(id: string): void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  const active = info?.models.find((m) => m.id === info.active);
  const fault = !!error || (active && !active.available);
  const dot = fault ? "dot-fault" : info?.loaded ? "dot-ok" : "dot-wait";
  const text = error ? "Server unreachable" : !info ? "Connecting…" : active?.label ?? "";
  const status = error ? "server unreachable" : !info ? "connecting" : fault ? "fault" : info.loaded ? `ready on ${info.device}` : "loading";
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button className="chip-btn" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        aria-label={`Model: ${text}, ${status}`} title={status}>
        <span className={`dot ${dot}`} aria-hidden="true" />
        <span>{text}</span>
      </button>
      <AnimatePresence>
        {open && info && (
          <motion.div className="popover" role="listbox" aria-label="Model"
            initial={{ opacity: 0, y: -4, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.14 }}>
            {info.models.map((m) => (
              <button key={m.id} role="option" aria-selected={m.id === info.active} className="popover-row" disabled={!m.available}
                onClick={() => { onSelect(m.id); setOpen(false); }}>
                <span className={`dot ${m.available ? (m.id === info.active ? "dot-ok" : "dot-wait") : "dot-fault"}`} style={{ marginTop: 7 }} />
                <span>
                  <strong>{m.label}</strong>{m.is_main ? " (main model)" : ""}
                  <small>{m.description}</small>
                  <small>Trained on {m.trained_on}</small>
                  {!m.available && <small style={{ color: "var(--fault)" }}>{m.reason}</small>}
                </span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
