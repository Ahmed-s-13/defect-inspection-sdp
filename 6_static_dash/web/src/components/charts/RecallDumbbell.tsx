import { motion } from "motion/react";
import { className, pct } from "../../lib/format";
import type { Reveal } from "../../lib/useReveal";

const ease = [0.22, 1, 0.36, 1] as const;

/** Per-class recall: grey dot = previous model on the same split, ink dot = this model.
 *  On reveal the rows arrive in turn and each ink dot slides from the previous model's value to its own. */
export function RecallDumbbell({ rows, reveal }: { rows: { cls: string; now: number; before: number | null }[]; reveal?: Reveal }) {
  const shown = reveal?.shown ?? true, t = reveal?.t ?? (() => 0);
  return (
    <div className="dumbbell" role="table" aria-label="Recall per class, this model against the previous one">
      <div className="db-scale" aria-hidden="true">{[0, 25, 50, 75, 100].map((v) => <span key={v} style={{ left: `${v}%` }}>{v}%</span>)}</div>
      {rows.map((r, i) => {
        const lo = Math.min(r.now, r.before ?? r.now), hi = Math.max(r.now, r.before ?? r.now);
        const delta = r.before == null ? null : r.now - r.before;
        const from = r.before == null ? 0 : (r.before - r.now) * 100; // % of the track
        const d = t(0.12) * i;
        return (
          <motion.div key={r.cls} className="db-row" role="row"
            initial={false} animate={{ opacity: shown ? 1 : 0, y: shown ? 0 : 12 }} transition={{ duration: t(0.45), delay: d, ease }}>
            <span className="db-label" role="rowheader">{className(r.cls)}</span>
            <span className="db-track" role="cell">
              <span className="db-link" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
              {r.before != null && <span className="db-dot db-before" style={{ left: `${r.before * 100}%` }} title={`Previous ${pct(r.before)}`} />}
              <motion.span className="db-slide" initial={false} animate={{ x: shown ? "0%" : `${from}%` }} transition={{ duration: t(0.8), delay: d + t(0.3), ease }}>
                <span className="db-dot db-now" style={{ left: `${r.now * 100}%` }} />
              </motion.span>
            </span>
            <span className="db-fig fig" role="cell">{pct(r.now)}{delta != null && <small className={delta >= 0 ? "up" : "down"}> {delta >= 0 ? "+" : "−"}{Math.abs(delta * 100).toFixed(1)}</small>}</span>
          </motion.div>
        );
      })}
    </div>
  );
}
