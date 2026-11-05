import type React from "react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { className, pct } from "../../lib/format";
import { ABOUT, PIPE, type Outcome } from "../PipelineScene";

/** The aircraft-or-not check as a picture: every point is a crop, placed by its AoN score (across) and the features'
 *  main other direction (down, for spread only). Aircraft-skin training crops sit left of the boundary, the non-aircraft
 *  photos right. Groups can be switched off; hovering names a point; the five pipeline crops are shown as pictures. */
export interface AonData {
  skin: [number, number, number][]; off: [number, number, number, number][]; categories: string[]; classes: string[];
  stop_logit: number; pipeline: { outcome: string; image: string; xy: [number, number] }[];
}
type Group = "skin" | "fit" | "held";
const GROUPS: [Group, string][] = [["skin", "Aircraft skin: the model's training crops"], ["fit", "Non-aircraft, training categories (people, faces, tools, vehicles, factories)"],
  ["held", "Non-aircraft, test categories, left out for testing (helmets, hands, gloves, hangars, runways)"]];
const W = 760, H = 420, PAD = 28;
const OUT_WORDS: Record<string, string> = { show: "Defect detected", ambiguous: "Ambiguous", possible: "Possible: check", nothing: "Nothing shown", reaim: "Re-aim" };
/** Each dot flies in from a distance as the chart reveals (and back out when it hides): from a point pushed away from
 *  the chart's centre, small, then to its place. */
const fly = (p: { x: number; y: number }, shown: boolean, i: number): React.CSSProperties => {
  const dx = (p.x - W / 2) * 1.6, dy = (p.y - H / 2) * 1.6, d = ((i * 37) % 100) / 100;
  return { transform: shown ? "none" : `translate(${dx}px, ${dy}px) scale(.2)`, transformBox: "fill-box", transformOrigin: "center",
    transition: `transform ${shown ? 0.9 : 0.4}s cubic-bezier(.22,1,.36,1) ${shown ? d * 0.6 : 0}s` };
};
const cat = (c: string) => c.replace(/^[a-z_]+\//, (m) => m).replace(/_/g, " ");

export function AonExplorer({ d, shown }: { d: AonData; shown: boolean }) {
  const [on, setOn] = useState<Record<Group, boolean>>({ skin: true, fit: true, held: true });
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);
  const [zoom, setZoom] = useState<string | null>(null);   // the pipeline crop shown large, by outcome
  useEffect(() => {
    if (!zoom) return;
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setZoom(null); };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [zoom]);
  const svg = useRef<SVGSVGElement>(null);
  const { sx, sy } = useMemo(() => {
    const all = [...d.skin.map((p) => [p[0], p[1]]), ...d.off.map((p) => [p[0], p[1]])];
    const x0 = Math.min(...all.map((p) => p[0])), x1 = Math.max(...all.map((p) => p[0]));
    const y0 = Math.min(...all.map((p) => p[1])), y1 = Math.max(...all.map((p) => p[1]));
    return { sx: (v: number) => PAD + ((v - x0) / (x1 - x0)) * (W - 2 * PAD), sy: (v: number) => PAD + ((y1 - v) / (y1 - y0)) * (H - 2 * PAD - 18) };
  }, [d]);
  const pts = useMemo(() => [
    ...d.skin.map((p) => ({ x: sx(p[0]), y: sy(p[1]), g: "skin" as Group, text: `Aircraft skin (${className(d.classes[p[2]])}), score ${p[0].toFixed(1)}` })),
    ...d.off.map((p) => ({ x: sx(p[0]), y: sy(p[1]), g: (p[3] ? "held" : "fit") as Group, text: `${cat(d.categories[p[2]])} (${p[3] ? "test category" : "training category"}), score ${p[0].toFixed(1)}` })),
  ], [d, sx, sy]);
  const onMove = (e: React.PointerEvent) => {
    const r = svg.current!.getBoundingClientRect(), mx = ((e.clientX - r.left) / r.width) * W, my = ((e.clientY - r.top) / r.height) * H;
    let best = null as null | (typeof pts)[number], bd = 64;
    for (const p of pts) { if (!on[p.g]) continue; const dd = (p.x - mx) ** 2 + (p.y - my) ** 2; if (dd < bd) { bd = dd; best = p; } }
    setHover(best ? { x: best.x, y: best.y, text: best.text } : null);
  };
  const delay = (g: Group) => (g === "skin" ? 0 : g === "fit" ? 0.35 : 0.7);
  return (
    <figure className="aon-fig">
      <div className="aon-toggles" role="group" aria-label="Show groups">
        {GROUPS.map(([g, label]) => (
          <button key={g} type="button" className={`aon-toggle aon-${g}`} aria-pressed={on[g]} onClick={() => setOn({ ...on, [g]: !on[g] })}><i aria-hidden="true" />{label}</button>
        ))}
      </div>
      <svg ref={svg} className="chart aon-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Scatter of aircraft-skin and non-aircraft crops along the aircraft-or-not score"
        onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {(["skin", "fit", "held"] as Group[]).map((g) => (
          <g key={g} className={`aon-pts aon-${g}`} style={{ opacity: shown && on[g] ? 1 : 0, transition: `opacity .6s ${shown ? delay(g) : 0}s` } as React.CSSProperties}>
            {pts.filter((p) => p.g === g).map((p, i) => <circle key={i} cx={p.x} cy={p.y} r="2.4" style={fly(p, shown, i)} />)}
          </g>
        ))}
        <line className="aon-boundary" x1={sx(0)} x2={sx(0)} y1={PAD - 10} y2={H - 30} />
        <text className="tick" x={sx(0) + 6} y={PAD - 2}>decision boundary: score 0 (P = 0.5)</text>
        <line className="aon-stop" x1={sx(d.stop_logit)} x2={sx(d.stop_logit)} y1={PAD + 14} y2={H - 30} />
        <text className="tick aon-stop-label" x={sx(d.stop_logit) + 6} y={PAD + 26}>conformal stop line</text>
        {d.pipeline.map((p) => (
          <g key={p.outcome} className="aon-crop" role="button" tabIndex={0} aria-label={`Show the ${OUT_WORDS[p.outcome]} crop large`}
            style={{ opacity: shown ? 1 : 0, transition: `opacity .5s ${shown ? 1.1 : 0}s` }}
            onClick={() => setZoom(p.outcome)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setZoom(p.outcome); } }}
            onPointerEnter={() => setHover({ x: sx(p.xy[0]), y: sy(p.xy[1]), text: `${OUT_WORDS[p.outcome]}: click to enlarge` })}>
            <image href={p.image} x={sx(p.xy[0]) - 13} y={sy(p.xy[1]) - 13} width="26" height="26" />
            <rect x={sx(p.xy[0]) - 13} y={sy(p.xy[1]) - 13} width="26" height="26" />
          </g>
        ))}
        <text className="tick" x={PAD} y={H - 6}>← aircraft skin</text>
        <text className="tick" x={W - PAD} y={H - 6} textAnchor="end">not aircraft skin →</text>
        {hover && <g className="aon-tip" transform={`translate(${Math.min(hover.x + 10, W - 250)} ${Math.max(hover.y - 28, 12)})`}>
          <rect width="244" height="22" rx="3" /><text x="8" y="15">{hover.text}</text></g>}
      </svg>
      <AnimatePresence>{zoom && <Zoom outcome={zoom as Outcome} onClose={() => setZoom(null)} />}</AnimatePresence>
      <figcaption className="chart-key">Across: the check's score (weights × standardised features + bias). Down: the features' main other direction, for spread only. Squares: the five crops from "Follow a crop"; click one to see it large.</figcaption>
    </figure>
  );
}

/** One pipeline crop in the middle of the screen, with what it is and what each stage made of it. */
function Zoom({ outcome, onClose }: { outcome: Outcome; onClose(): void }) {
  const c = PIPE.crops.find((x) => x.outcome === outcome)!;
  const off = c.truth === "off-target";
  const words = c.set.filter((x) => x !== "no-defect").map((x) => className(x).toLowerCase()).join(" or ");
  const answer = outcome === "show" ? `${className(c.set[0])} ${pct(c.tempered[PIPE.classes.indexOf(c.set[0])], 0)}` : outcome === "ambiguous" ? `Ambiguous: ${words}`
    : outcome === "possible" ? `Possible ${words}: check` : outcome === "nothing" ? "Nothing shown" : "Can't assess: re-aim";
  return (
    <motion.div className="aon-modal" role="dialog" aria-modal="true" aria-label={ABOUT[outcome]} onClick={onClose}
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
      <motion.figure onClick={(e) => e.stopPropagation()} initial={{ scale: 0.94, y: 12 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.96, opacity: 0 }}
        transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}>
        <button type="button" className="aon-close" aria-label="Close" onClick={onClose}>×</button>
        <img src={c.image} alt={ABOUT[outcome]} />
        <figcaption>
          <strong>{off ? "A glove: not aircraft skin" : `A test crop: ${ABOUT[outcome]}`}</strong>
          <span>{off ? <>An Open Images photo from the team's non-aircraft set, cut to the model's 224 px square{c.credit ? <> (photo: {c.credit.author}, {c.credit.license})</> : null}.</>
            : <>A 224 px square from the test photos, never trained or calibrated on. True class: {className(c.truth).toLowerCase()}.</>}</span>
          <span>Aircraft-or-not: {pct(c.aon_score, 1)} "not aircraft", p = {c.p_aon.toFixed(3)}; with the kNN distance, combined p = {c.p_ood.toFixed(3)} (stops at {PIPE.served.ood} or below).</span>
          <span>Architecture B shows: <b>{answer}</b></span>
        </figcaption>
      </motion.figure>
    </motion.div>
  );
}
