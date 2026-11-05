import type React from "react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Ranked, StateId } from "../api/client";
import { className, int, pct } from "../lib/format";
import { useReveal } from "../lib/useReveal";
import { RuleDraw } from "./RuleDraw";
import { SoftmaxBars } from "./SoftmaxBars";
import { STATE_WORDS, StateGlyph } from "./StateBadge";

/** scripts/sweep_crops.py: architecture B's answer for every 224 px square on one photo, 16 px apart. */
interface Sweep {
  photo: string; w: number; h: number; side: number; stride: number; cols: number; rows: number;
  classes: string[]; states: StateId[]; set_threshold: number;
  cells: [number, number[], number, number | null, number | null][]; // state, probs in 1/1000, answer bitmask, p_quality, p_ood
}
interface Answer { x: number; y: number; state: StateId; ranked: Ranked[]; answer: string[]; top: string; confidence: number; pOod: number | null }

const DIR = "sweep/";
/** Where each stop reads best, as a fraction of the scroll through the scene; presentation mode stops here. */
export const SQUARE_AT = [0.02, 0.24, 0.31, 0.455, 0.625, 0.79, 0.97];

// the plane is drawn in the photo's own frame (1,024 x 1,240, nose up); the photo sits on the right wing here
const PW = 1024, PH = 1240;
const SPOT = { x: 790, y: 684, w: 60 };
const SPOT_H = SPOT.w * PH / PW;
// the square's stops (top-left corner, photo pixels) and the scroll fraction it holds each one over
const PATH: [number, number, number][] = [
  [0.27, 64, 112], [0.345, 64, 112],     // plain paint
  [0.43, 224, 304], [0.48, 224, 304],    // on the crack
  [0.545, 368, 624],                     // past the hole
  [0.6, 320, 832], [0.65, 320, 832],     // where the crack meets the bent panel
  [0.755, 608, 736], [1, 608, 736],      // on the bent panel
];
const CAPTIONS: [string, string][] = [
  ["The model sees a square, not the plane", "On the helmet, the inspector aims a 224 px square. The model answers only for what is inside it."],
  ["A real photo, close up", "A cracked, punctured skin panel from D1. The model trained on it only as one square of the whole panel shrunk to 224 px, never at this range."],
  ["Plain paint: No defect", "Intact skin now has a class of its own. The 6-class model could only refuse this square; this one says the paint is undamaged."],
  ["On the crack: Crack", "Over the crack lines and the hole, the answer is one class."],
  ["Where they meet: Ambiguous", "Dent also passes the cut-off here, so two classes are kept and a person checks."],
  ["On the bent panel: Dent", "The same photo, a different answer: a different part of it is in view."],
  ["Every square at once", "We scored all the squares, 16 px apart. Each cell is the answer for the square centred there: Crack follows the cracks, Dent the middle of the bent panel, plain paint is No defect, and the edges of the damage are Ambiguous."],
];
const PHASE_AT = [0.13, 0.27, 0.39, 0.565, 0.7, 0.85]; // caption boundaries

const ease3 = (t: number) => 1 - Math.pow(1 - t, 3);
const smooth = (t: number) => t * t * (3 - 2 * t);
const clamp = (v: number) => Math.max(0, Math.min(1, v));

function along(p: number): [number, number] {
  if (p <= PATH[0][0]) return [PATH[0][1], PATH[0][2]];
  for (let i = 1; i < PATH.length; i++) {
    const [p1, x1, y1] = PATH[i], [p0, x0, y0] = PATH[i - 1];
    if (p <= p1) { const t = smooth((p - p0) / (p1 - p0)); return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]; }
  }
  const last = PATH[PATH.length - 1];
  return [last[1], last[2]];
}

function answerAt(s: Sweep, col: number, row: number): Answer {
  const [st, pm, mask, , pOod] = s.cells[row * s.cols + col];
  const probs = pm.map((v) => v / 1000);
  const ranked = s.classes.map((cls, i) => ({ cls, p: probs[i] })).sort((a, b) => b.p - a.p);
  return { x: col * s.stride, y: row * s.stride, state: s.states[st], ranked, answer: s.classes.filter((_, i) => mask >> i & 1),
    top: ranked[0].cls, confidence: ranked[0].p, pOod };
}

// the answer map: one colour per single class, then the two kinds of refusal
const MAP_COLOR: Record<string, string> = { crack: "var(--detected)", dent: "var(--map-dent)" };
const mapKind = (s: Sweep, c: Sweep["cells"][number]) => {
  const st = s.states[c[0]];
  if (st !== "detected") return st;
  return s.classes[c[1].indexOf(Math.max(...c[1]))];
};
const kindColor = (k: string) => MAP_COLOR[k] ?? (k === "clear" ? "var(--clear)" : k === "ambiguous" || k === "unclear" ? "var(--verify)" : k === "cannot_assess" || k === "retake" ? "var(--unfamiliar)" : "var(--ink-2)");
const kindAlpha = (k: string) => (k === "clear" ? 0.22 : k === "ambiguous" || k === "unclear" ? 0.3 : k === "cannot_assess" || k === "retake" ? 0.42 : 0.62); // defect classes stand out
const kindOrder = (k: string) => (k === "clear" ? 1 : k === "ambiguous" || k === "unclear" ? 2 : k === "cannot_assess" || k === "retake" ? 3 : 0);
const kindLabel = (k: string) => (k in STATE_WORDS && k !== "detected" ? STATE_WORDS[k as StateId] : className(k));

/** A drawn plane zooms into one wing panel, a real photo of it, then the model's square slides across that photo
 *  while the answer changes underneath; it ends on the answer for every square position at once. */
export function SquareScene({ first, dimAt }: { first: number; dimAt(step: number): boolean }) {
  const [sweep, setSweep] = useState<Sweep | null>(null);
  useEffect(() => { fetch(DIR + "sweep.json").then((r) => r.json()).then(setSweep).catch(() => setSweep(null)); }, []);
  const run = useRef<HTMLDivElement>(null), stage = useRef<HTMLDivElement>(null), plane = useRef<SVGSVGElement>(null);
  const photo = useRef<HTMLDivElement>(null), marker = useRef<SVGGElement>(null), square = useRef<HTMLDivElement>(null);
  const shade = useRef<SVGPathElement>(null), mapClip = useRef<SVGRectElement>(null), mapG = useRef<SVGGElement>(null);
  const [phase, setPhase] = useState(0);
  const [cell, setCell] = useState<[number, number]>([4, 7]);
  const rv = useReveal<HTMLDivElement>(0.9);

  useLayoutEffect(() => {
    if (!sweep) return;
    const runEl = run.current!, st = stage.current!;
    const bar = document.querySelector<HTMLElement>(".topbar")?.offsetHeight ?? 56;
    st.style.setProperty("--stick", `${bar}px`);
    let raf = 0, lastPhase = -1, lastCell = "";
    const frame = () => {
      raf = 0;
      const span = runEl.offsetHeight - st.offsetHeight;
      const p = clamp((st.getBoundingClientRect().top - runEl.getBoundingClientRect().top) / span);
      // camera: from the whole plane to the photo's patch of wing, scale changing evenly
      const z = ease3(clamp((p - 0.07) / 0.15));
      const w = PW * Math.pow(SPOT.w / PW, z), h = w * PH / PW, f = (PW - w) / (PW - SPOT.w);
      const cx = PW / 2 + (SPOT.x - PW / 2) * f, cy = PH / 2 + (SPOT.y - PH / 2) * f;
      const camX = cx - w / 2, camY = cy - h / 2;
      plane.current!.setAttribute("viewBox", `${camX} ${camY} ${w} ${h}`);
      marker.current!.style.opacity = String(1 - clamp(z / 0.25));
      const ph = photo.current!;
      ph.style.transform = `translate(${((SPOT.x - SPOT.w / 2 - camX) / w) * 100}%, ${((SPOT.y - SPOT_H / 2 - camY) / h) * 100}%) scale(${SPOT.w / w})`;
      const fade = clamp((z - 0.72) / 0.28);
      ph.style.opacity = String(fade);
      ph.style.setProperty("--m", String(fade)); // the edges resolve last
      // the square, snapped to the 16 px grid it was scored on
      const [x, y] = along(p);
      const col = Math.min(sweep.cols - 1, Math.round(x / sweep.stride)), row = Math.min(sweep.rows - 1, Math.round(y / sweep.stride));
      const sx = col * sweep.stride, sy = row * sweep.stride, side = sweep.side;
      const on = clamp((p - 0.255) / 0.025) * (1 - clamp((p - 0.85) / 0.035));
      Object.assign(square.current!.style, { left: `${(sx / sweep.w) * 100}%`, top: `${(sy / sweep.h) * 100}%`, opacity: String(on) });
      shade.current!.setAttribute("d", `M0 0H${sweep.w}V${sweep.h}H0Z M${sx} ${sy}v${side}h${side}v${-side}Z`);
      shade.current!.style.opacity = String(on * 0.5);
      if (`${col},${row}` !== lastCell) { lastCell = `${col},${row}`; setCell([col, row]); }
      // the map fills in top to bottom
      const m = clamp((p - 0.86) / 0.09);
      mapClip.current!.setAttribute("height", String(m * sweep.h));
      mapG.current!.style.opacity = String(clamp((p - 0.86) / 0.02));
      const phase = PHASE_AT.findIndex((b) => p < b);
      const next = phase < 0 ? PHASE_AT.length : phase;
      if (next !== lastPhase) { lastPhase = next; setPhase(next); }
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(frame); };
    frame();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, [sweep]);

  const a = sweep ? answerAt(sweep, ...cell) : null;
  const kinds = useMemo(() => {
    if (!sweep) return [];
    const n = new Map<string, number>();
    sweep.cells.forEach((c) => { const k = mapKind(sweep, c); n.set(k, (n.get(k) ?? 0) + 1); });
    return [...n.entries()].sort((x, y) => kindOrder(x[0]) - kindOrder(y[0]) || y[1] - x[1]);
  }, [sweep]);
  const dim = SQUARE_AT.every((_, i) => dimAt(first + i));

  return (
    <div className="panel sq-sec" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title">Where the square sits changes the answer</h2>
      <p className="panel-sub">The model answers for one 224 px square, never the whole photo, so the inspector can point at a defect and ask about it. Here is the real model on one photo.</p>
      <div className="scene-run sq-run present-dim" data-dim={dim} ref={run} aria-label="A plane zooms into a wing panel; the model's square moves across a photo of it and its answer changes">
        {SQUARE_AT.map((at, i) => <i key={at} className="stage-mark" data-step={first + i} data-step-at={at} aria-hidden="true" />)}
        <div className="scene sq-stage" ref={stage}>
          <div className="sq-frame">
            <svg ref={plane} className="sq-plane" viewBox={`0 0 ${PW} ${PH}`} aria-hidden="true">
              <Plane />
              <g ref={marker}>
                <circle className="sq-ring" cx={SPOT.x} cy={SPOT.y} r="44" vectorEffect="non-scaling-stroke" />
                <path className="sq-lead" d={`M${SPOT.x + 31} ${SPOT.y + 31} L${SPOT.x + 84} ${SPOT.y + 130}`} vectorEffect="non-scaling-stroke" />
                <text className="sq-callout" x={SPOT.x + 84} y={SPOT.y + 166} textAnchor="middle">this panel</text>
              </g>
            </svg>
            <div className="sq-photo" ref={photo}>
              <img src={DIR + "photo.jpg"} alt="A cracked, punctured aircraft skin panel (D1 photo aircraft_000139)" />
              <svg viewBox={`0 0 ${sweep?.w ?? PW} ${sweep?.h ?? PH}`} aria-hidden="true">
                <defs><clipPath id="sq-map-clip"><rect ref={mapClip} x="0" y="0" width={sweep?.w ?? PW} height="0" /></clipPath></defs>
                <g ref={mapG} className="sq-map" clipPath="url(#sq-map-clip)" shapeRendering="crispEdges">
                  {sweep?.cells.map((c, k) => (
                    <rect key={k} x={(k % sweep.cols) * sweep.stride + sweep.side / 2 - sweep.stride / 2} y={Math.floor(k / sweep.cols) * sweep.stride + sweep.side / 2 - sweep.stride / 2}
                      width={sweep.stride} height={sweep.stride} fill={kindColor(mapKind(sweep, c))} fillOpacity={kindAlpha(mapKind(sweep, c))} />
                  ))}
                </g>
                <path ref={shade} className="sq-shade" fillRule="evenodd" />
              </svg>
              <div ref={square} className={`reticle sq-square${a ? ` tone-${a.state}` : ""}`} style={sweep ? { width: `${(sweep.side / sweep.w) * 100}%`, height: `${(sweep.side / sweep.h) * 100}%` } : undefined} aria-hidden="true">
                <span className="bracket bracket-nw" /><span className="bracket bracket-ne" /><span className="bracket bracket-sw" /><span className="bracket bracket-se" />
                <span className="reticle-label">Model's view 224 px</span>
              </div>
            </div>
          </div>

          <div className="sq-side">
            <p className="sq-count" aria-hidden="true"><span className="step-n">{phase + 1}</span> of {CAPTIONS.length}</p>
            <div className="sq-caption" aria-live="polite">
              {CAPTIONS.map(([head, text], i) => (
                <div key={head} className={phase === i ? "on" : undefined}><strong>{head}</strong><p>{text}</p></div>
              ))}
            </div>
            <div className={`sq-readout${phase >= 2 && phase <= 5 && a ? " on" : ""}`}>
              {a && (
                <>
                  <h3 className={`verdict verdict-${a.state}`}>
                    <span className="verdict-state"><StateGlyph state={a.state} size={22} /> {STATE_WORDS[a.state]}</span>
                    {a.answer.length > 0 && <span className={`verdict-class${a.answer.length > 1 ? " is-set" : ""}`}>
                      {a.state === "clear" ? "Intact surface" : a.answer.map(className).join(" or ")}
                      {(a.state === "detected" || a.state === "clear") && <span className="fig verdict-pct"> {pct(a.confidence)}</span>}
                    </span>}
                  </h3>
                  {a.answer.length === 0 && <p className="verdict-sub">{a.state === "retake" ? "Re-take the photo." : "Re-aim: this looks unlike any training photo."}</p>}
                  <SoftmaxBars ranked={a.ranked} state={a.state} answer={a.answer} threshold={sweep!.set_threshold} />
                  <p className="sq-where">Square at <span className="fig">{a.x}, {a.y}</span> px{a.pOod != null && <> · unfamiliar-input check p = <span className="fig">{a.pOod.toFixed(2)}</span></>}</p>
                </>
              )}
            </div>
            <ul className={`sq-legend${phase === 6 ? " on" : ""}`}>
              {kinds.map(([k, n]) => (
                <li key={k}><i style={{ background: kindColor(k), opacity: kindAlpha(k) + 0.25 } as React.CSSProperties} /><span>{kindLabel(k)}</span><span className="fig">{int(n)}</span><span className="fig sub">{n / (sweep?.cells.length ?? 1) < 0.005 ? "<1%" : pct(n / (sweep?.cells.length ?? 1), 0)}</span></li>
              ))}
              {sweep && <li className="sq-legend-total"><span>Squares scored</span><span className="fig">{int(sweep.cells.length)}</span></li>}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

/** A plain top view of an airliner, nose up, in the photo's 1,024 x 1,240 frame. */
function Plane() {
  const s = { vectorEffect: "non-scaling-stroke" } as const;
  return (
    <g className="sq-air">
      <path {...s} d="M456 470 L80 742 L72 774 L266 724 L456 700 Z" />
      <path {...s} d="M568 470 L944 742 L952 774 L758 724 L568 700 Z" />
      <path {...s} d="M456 1010 L262 1110 L260 1134 L456 1092 Z" />
      <path {...s} d="M568 1010 L762 1110 L764 1134 L568 1092 Z" />
      <rect {...s} x="326" y="518" width="38" height="112" rx="16" />
      <rect {...s} x="660" y="518" width="38" height="112" rx="16" />
      <path {...s} d="M512 60 C548 60 566 120 568 200 L568 1040 C568 1100 540 1168 512 1180 C484 1168 456 1100 456 1040 L456 200 C458 120 476 60 512 60 Z" />
      <path {...s} className="sq-detail" d="M506 990 L518 990 L516 1176 L508 1176 Z" />
      <path {...s} className="sq-detail" d="M486 150 Q512 136 538 150" />
      {/* flaps, ailerons and the panel seams the camera passes through */}
      <path {...s} className="sq-detail" d="M600 703 L758 716 M758 716 L900 760 M436 703 L266 716 M266 716 L124 760" />
      <path {...s} className="sq-detail" d="M700 600 L716 717 M760 640 L772 720 M820 683 L826 735 M870 719 L874 750 M324 600 L308 717 M264 640 L252 720" />
      <path {...s} className="sq-detail" d="M620 540 L900 744 M404 540 L124 744" />
    </g>
  );
}
