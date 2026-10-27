import { motion } from "motion/react";
import { className, colClass, shortClass } from "../lib/format";
import type { Reveal } from "../lib/useReveal";

/** Rows are the true class, columns the predicted class. Cell darkness is the row share; hover shows counts. */
export function ConfusionMatrix({ classes, matrix, normalise = true, selected, onSelect, label, reveal }: {
  classes: string[]; matrix: number[][]; normalise?: boolean; selected?: number | null; onSelect?(i: number | null): void; label: string;
  reveal?: Reveal; // on reveal the diagonal (correct calls) fills first, then the errors
}) {
  const shown = reveal?.shown ?? true, t = reveal?.t ?? (() => 0);
  const n = classes.length;
  const rowSum = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  return (
    <div className="cm" role="table" aria-label={label} style={{ gridTemplateColumns: `minmax(7em, auto) repeat(${classes.length}, minmax(40px, 1fr))` }}>
      <div className="cm-corner" role="columnheader">True ↓ · Called →</div>
      {classes.map((c) => <div key={c} className="cm-col" role="columnheader" title={className(c)}>{colClass(c)}</div>)}
      {matrix.map((row, i) => (
        <div key={i} role="row" className={`cm-row${selected === i ? " is-selected" : ""}${selected != null && selected !== i ? " is-dim" : ""}`} style={{ display: "contents" }}>
          <button className="cm-rowhead" role="rowheader" onClick={() => onSelect?.(selected === i ? null : i)} aria-pressed={selected === i}>
            {shortClass(classes[i])}<span className="cm-n fig"> {rowSum[i]}</span>
          </button>
          {row.map((v, j) => {
            const share = rowSum[i] ? v / rowSum[i] : 0;
            const shownValue = normalise ? share : v;
            return (
              <div key={j} role="cell" className={`cm-cell${i === j ? " is-diag" : ""}`} title={`${shortClass(classes[i])} called ${shortClass(classes[j])}: ${v} of ${rowSum[i]}`}>
                <motion.span className="cm-fill" initial={false} animate={{ opacity: shown ? share * 0.9 : 0 }}
                  transition={reveal ? { duration: t(0.45), delay: i === j ? t(0.07) * i : t(0.07) * n + t(0.25) + t(0.025) * (i + j) } : { duration: 0.3 }} />
                <motion.span className={`cm-val fig${share > 0.5 ? " on-dark" : ""}`} initial={false} animate={{ opacity: shown ? 1 : 0 }}
                  transition={{ duration: t(0.35), delay: i === j ? t(0.07) * i : t(0.07) * n + t(0.25) + t(0.025) * (i + j) }}>{v === 0 ? "·" : normalise ? Math.round(shownValue * 100) : v}</motion.span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
