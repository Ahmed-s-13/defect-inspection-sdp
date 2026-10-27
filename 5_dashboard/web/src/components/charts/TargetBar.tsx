import { motion } from "motion/react";
import type { Reveal } from "../../lib/useReveal";

const ease = [0.22, 1, 0.36, 1] as const;

/** Where macro-F1 sits against the 0.85 target, with its 95% interval. Scale 0.5–1.0.
 *  On reveal the interval opens out from the value, then the target tick slides in: the gap being measured. */
export function TargetBar({ value, ci, target, reveal }: { value: number; ci: [number, number] | null; target: number; reveal?: Reveal }) {
  const x = (v: number) => `${((v - 0.5) / 0.5) * 100}%`;
  const shown = reveal?.shown ?? true, t = reveal?.t ?? (() => 0);
  const origin = ci ? `${((value - ci[0]) / (ci[1] - ci[0])) * 100}% 50%` : "50% 50%";
  return (
    <div className="target" role="img" aria-label={`Macro-F1 ${value.toFixed(3)} against a target of ${target}`}>
      <div className="target-track">
        {ci && <motion.span className="target-ci" style={{ left: x(ci[0]), width: `calc(${x(ci[1])} - ${x(ci[0])})`, transformOrigin: origin }}
          initial={false} animate={{ scaleX: shown ? 1 : 0 }} transition={{ duration: t(0.7), delay: t(0.25), ease }} />}
        <span className="target-value" style={{ left: x(value) }} />
        <motion.span className="target-tick" style={{ left: x(target) }}
          initial={false} animate={{ x: shown ? 0 : -48, opacity: shown ? 1 : 0 }} transition={{ duration: t(0.6), delay: t(0.85), ease }} />
      </div>
      <div className="target-scale"><span>0.50</span>
        <motion.span style={{ left: x(target) }} className="target-label"
          initial={false} animate={{ opacity: shown ? 1 : 0 }} transition={{ duration: t(0.5), delay: t(1.1) }}>Our target {target.toFixed(2)}</motion.span>
        <span>1.00</span></div>
    </div>
  );
}
