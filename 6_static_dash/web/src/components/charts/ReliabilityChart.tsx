import { motion } from "motion/react";
import type { Bin } from "../../api/client";
import type { Reveal } from "../../lib/useReveal";

/** Reliability: each bar is how often the model was right at that stated confidence. The dashed line is "honest".
 *  `before`, when given, is drawn as dots on top (the raw percentages, before calibration).
 *  On reveal the bars rise left to right, then the honest line draws across them. */
export function ReliabilityChart({ bins, before, reveal, label }: { bins: Bin[]; before?: Bin[]; reveal?: Reveal; label: string }) {
  const W = 520, H = 300, L = 40, B = 28, T = 12, R = 10;
  const x = (v: number) => L + v * (W - L - R), y = (v: number) => T + (1 - v) * (H - T - B);
  const shown = reveal?.shown ?? true, t = reveal?.t ?? (() => 0);
  const kept = bins.filter((b) => b.n >= 20), ghost = (before ?? []).filter((b) => b.n >= 20);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={label}>
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="grid" />
          <text x={L - 6} y={y(v) + 4} textAnchor="end" className="tick">{v * 100}%</text>
          <text x={x(v)} y={H - 8} textAnchor="middle" className="tick">{v * 100}%</text>
        </g>
      ))}
      {kept.map((b, i) => (
        <motion.rect key={b.lo} x={x(b.lo) + 2} width={x(b.hi) - x(b.lo) - 4} y={y(b.accuracy)} height={y(0) - y(b.accuracy)} className="bar"
          style={{ transformBox: "fill-box", transformOrigin: "50% 100%" }}
          initial={false} animate={{ scaleY: shown ? 1 : 0 }} transition={{ duration: t(0.55), delay: t(0.08) * i, ease: [0.22, 1, 0.36, 1] }}>
          <title>{`Confidence ${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%: right ${Math.round(b.accuracy * 100)}% of ${b.n}`}</title>
        </motion.rect>
      ))}
      {ghost.map((b, i) => (
        <motion.circle key={`raw-${b.lo}`} cx={(x(b.lo) + x(b.hi)) / 2} cy={y(b.accuracy)} r={5} className="dot-before"
          initial={false} animate={{ opacity: shown ? 1 : 0, scale: shown ? 1 : 0.4 }}
          transition={{ duration: t(0.35), delay: t(0.08) * kept.length + t(0.06) * i }}>
          <title>{`Raw confidence ${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%: right ${Math.round(b.accuracy * 100)}% of ${b.n}`}</title>
        </motion.circle>
      ))}
      <motion.line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="ref"
        style={{ transformBox: "fill-box", transformOrigin: "0% 100%" }}
        initial={false} animate={{ scale: shown ? 1 : 0, opacity: shown ? 1 : 0 }}
        transition={{ duration: t(0.9), delay: t(0.08) * kept.length + t(0.2), ease: "easeInOut" }} />
    </svg>
  );
}
