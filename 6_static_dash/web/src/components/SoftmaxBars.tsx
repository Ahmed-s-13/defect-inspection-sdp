import type React from "react";
import { motion } from "motion/react";
import type { Ranked, StateId } from "../api/client";
import { className } from "../lib/format";
import { AnimatedNumber } from "./AnimatedNumber";

/** Rows keep their identity across predictions, so values and order morph from the previous result.
 *  With the calibrated layer (architecture B), the tick on each track is the set's cut-off: every class past it is in the answer.
 *  When a gate stopped the crop, the bars stay as a faint record of what the model would have said. */
export function SoftmaxBars({ ranked, state, answer, threshold }: {
  ranked: Ranked[]; state: StateId; answer: string[]; threshold: number | null;
}) {
  const gated = state === "retake" || state === "cannot_assess";
  const inSet = (cls: string) => (state === "unrated" ? cls === ranked[0].cls : answer.includes(cls));
  return (
    <div className={`bars-wrap${gated ? " is-gated" : ""}`}>
      {gated && <p className="bars-note">Not shown to the inspector. What the model would have said:</p>}
      <ol className="bars" aria-label="Calibrated probability for each defect class" style={threshold != null ? { "--cut": threshold } as React.CSSProperties : undefined}>
        {ranked.map((r) => (
          <motion.li layout="position" key={r.cls} className={`bar-row${inSet(r.cls) ? " is-in" : ""}`}
            transition={{ type: "spring", stiffness: 500, damping: 42 }}>
            <span className="bar-label">{className(r.cls)}</span>
            <span className="bar-track">
              <motion.span className={`bar-fill${inSet(r.cls) ? ` fill-${state}` : ""}`} initial={false}
                animate={{ scaleX: Math.max(r.p, 0.004) }} transition={{ type: "spring", stiffness: 240, damping: 30 }} />
              {threshold != null && <i className="bar-cut" aria-hidden="true" />}
            </span>
            <AnimatedNumber className="bar-figure" value={r.p * 100} format={(v) => `${v.toFixed(1)}%`} />
          </motion.li>
        ))}
      </ol>
      {threshold != null && !gated && <p className="bars-key"><i className="bar-cut-key" aria-hidden="true" /> Classes past {(threshold * 100).toFixed(1)}% are kept in the answer</p>}
    </div>
  );
}
