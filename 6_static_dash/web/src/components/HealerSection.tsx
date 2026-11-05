import type React from "react";
import { motion } from "motion/react";
import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import cc from "../data/cleancrop.json";
import healer from "../data/healer.json";
import { className, int, pct } from "../lib/format";
import { useReveal } from "../lib/useReveal";

/** scripts/healer_samples.py: three healed D1 photos and the counts of the healed "no defect" set. */
const H = healer;
const D = cc.decoys; // the LaMa check, from gc7's test report
const DIR = "healer/";
const vars = (v: Record<string, string | number>) => v as React.CSSProperties;
const ease = [0.22, 1, 0.36, 1] as const;
const ROUTE_WORDS: Record<string, string> = { "painted over": "painted over", "screw head put back": "screw head put back", "painted over, no head put back": "painted over (too small for a head)" };

/** "No defect" photos made by healing D1: a before/after slider over real samples, then how the healer works. */
export function HealerSection({ first, dimAt }: { first: number; dimAt(step: number): boolean }) {
  return (
    <section className="heal-sec">
      <h2 className="data-h2">Making photos with no damage</h2>
      <p className="how-sub heal-lede">The first source: real D1 photos with the labelled damage painted out. The skin around each defect then shows what that same spot looks like intact.</p>
      <Compare step={first} dim={dimAt(first)} />
      <HowItHeals step={first + 1} dim={dimAt(first + 1)} />
    </section>
  );
}

function Compare({ step, dim }: { step: number; dim: boolean }) {
  const [i, setI] = useState(0);
  const [pos, setPos] = useState(50);
  const [boxes, setBoxes] = useState(true);
  const frame = useRef<HTMLDivElement>(null), drag = useRef(false);
  const rv = useReveal<HTMLDivElement>(0.75);
  const s = H.samples[i];
  // on first sight the divider sweeps across once, so it reads as something to drag
  const swept = useRef(false);
  useEffect(() => {
    if (!rv.shown || swept.current || rv.t(1) === 0) return;
    swept.current = true;
    const t0 = performance.now();
    let raf = requestAnimationFrame(function tick(t) {
      const k = Math.min(1, (t - t0) / 1600), e = 1 - Math.pow(1 - k, 3);
      setPos(k < 1 ? 50 + 42 * Math.sin(e * Math.PI) : 50);
      if (k < 1 && !drag.current) raf = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(raf);
  }, [rv.shown]); // eslint-disable-line react-hooks/exhaustive-deps

  const at = (e: PointerEvent) => {
    const r = frame.current!.getBoundingClientRect();
    setPos(Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100)));
  };
  const key = (e: KeyboardEvent) => {
    const d = { ArrowLeft: -5, ArrowRight: 5, Home: -100, End: 100 }[e.key];
    if (d === undefined) return;
    e.preventDefault();
    setPos((p) => Math.max(0, Math.min(100, p + d)));
  };
  const routes = Object.entries(s.routes);

  return (
    <div className="heal-compare present-dim" data-step={step} data-dim={dim} ref={rv.ref} data-shown={rv.shown}>
      <div className="heal-frame rv" style={vars({ "--i": 0, aspectRatio: `${s.w} / ${s.h}` })} ref={frame}
        onPointerDown={(e) => { drag.current = true; (e.currentTarget as Element).setPointerCapture(e.pointerId); at(e); }}
        onPointerMove={(e) => { if (drag.current) at(e); }} onPointerUp={() => { drag.current = false; }} onPointerCancel={() => { drag.current = false; }}>
        <img src={`${DIR}g${s.group}_after.jpg`} alt={`The same photo healed: ${s.classes.map(className).join(" and ").toLowerCase()} removed`} draggable={false} />
        <div className="heal-before" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
          <img src={`${DIR}g${s.group}_before.jpg`} alt={`D1 photo with ${s.classes.map(className).join(" and ").toLowerCase()}`} draggable={false} />
          {boxes && (
            <svg viewBox={`0 0 ${s.w} ${s.h}`} aria-hidden="true">
              {s.boxes.map(([x1, y1, x2, y2], k) => <rect key={k} x={x1} y={y1} width={x2 - x1} height={y2 - y1} vectorEffect="non-scaling-stroke" />)}
            </svg>
          )}
        </div>
        <span className="heal-tag is-before" style={{ opacity: pos > 14 ? 1 : 0 }}>Before</span>
        <span className="heal-tag is-after" style={{ opacity: pos < 86 ? 1 : 0 }}>Healed</span>
        <div className="heal-divider" style={{ left: `${pos}%` }} role="slider" tabIndex={0} aria-label="Before and after: drag or use the arrow keys"
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pos)} onKeyDown={key}>
          <span className="heal-handle" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M9 6 3 12l6 6M15 6l6 6-6 6" /></svg></span>
        </div>
      </div>

      <div className="heal-side">
        <p className="heal-hint rv" style={vars({ "--i": 1 })}>Drag the divider. Left of it is the photo as D1 ships it, right of it the healed photo.</p>
        <div className="heal-samples rv" style={vars({ "--i": 2 })} role="radiogroup" aria-label="Sample photo">
          {H.samples.map((x, k) => (
            <button key={x.group} role="radio" aria-checked={k === i} className="heal-chip" onClick={() => { setI(k); setPos(50); }}>
              <img src={`${DIR}g${x.group}_before.jpg`} alt="" />
              <span>{x.classes.map(className).join(", ")}</span>
            </button>
          ))}
        </div>
        <motion.dl key={s.group} className="heal-facts" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease }}>
          <div><dt>Labelled boxes</dt><dd>{s.boxes.length} ({s.classes.map((c) => className(c).toLowerCase()).join(", ")})</dd></div>
          <div><dt>What the healer did</dt><dd>{routes.map(([r, n]) => `${n} ${ROUTE_WORDS[r] ?? r}`).join(", ")}</dd></div>
          <div><dt>Share of the photo repainted</dt><dd className="fig">{Math.round(s.healed_frac * 100)}%</dd></div>
        </motion.dl>
        <label className="heal-toggle"><input type="checkbox" checked={boxes} onChange={(e) => setBoxes(e.target.checked)} /> Show the labelled boxes</label>
      </div>
    </div>
  );
}

function HowItHeals({ step, dim }: { step: number; dim: boolean }) {
  const rv = useReveal<HTMLDivElement>(0.8);
  const a = H.auto as Record<string, number>, reviewedKept = H.kept - a.keep;
  const funnel: [string, number, string][] = [
    ["Healed crops", H.crops, `cut at the same squares as the defect crops (part 3 shows how), from ${int(H.photos)} photos, one per photo group`],
    ["Dropped by rules", a.drop, "too little real photo left, or a hole with no screw head put back"],
    ["Flagged and checked by hand", a.review, `odd colour, a visible seam or a smeared fill; we kept ${int(reviewedKept)}`],
    ["Kept as \"no defect\"", H.kept, `${int(H.kept_split.train)} train, ${int(H.kept_split.val)} validation, ${int(H.kept_split.test)} test, each in its photo group's split`],
  ];
  return (
    <div className="heal-how present-dim" data-step={step} data-dim={dim} ref={rv.ref} data-shown={rv.shown}>
      <h3 className="heal-h3 rv" style={vars({ "--i": 0 })}>How the healer works</h3>
      <ol className="heal-steps">
        <li className="rv" style={vars({ "--i": 1 })}>
          <strong><span className="step-n">1</span>Paint the damage out</strong>
          <p>Every labelled box, grown by 12% on each side, is masked. LaMa, a pretrained inpainting network, fills the mask with skin that continues its surroundings. Its Fourier convolutions see the whole photo at once, so panel lines and paint carry through the fill. {int(H.routes.lama + H.routes["lama only"])} boxes were painted over.</p>
        </li>
        <li className="rv" style={vars({ "--i": 2 })}>
          <strong><span className="step-n">2</span>Put screw heads back</strong>
          <p>A filled hole is not what an intact fastener looks like. For a missing head, an intact head is copied from the same photo (or from a bank of 16 taken from training photos), scaled to the hole and blended in. {int(H.routes.head)} heads were put back; holes too small to rebuild, in distant shots, were not used.</p>
        </li>
        <li className="rv" style={vars({ "--i": 3 })}>
          <strong><span className="step-n">3</span>Keep only convincing crops</strong>
          <ol className="heal-funnel">
            {funnel.map(([label, n, note], k) => (
              <li key={label} className={k === funnel.length - 1 ? "is-kept" : undefined}>
                <span className="heal-funnel-row"><span>{label}</span><span className="fig">{int(n)}</span></span>
                <span className="heal-funnel-track"><motion.i initial={false} animate={{ scaleX: rv.shown ? n / H.crops : 0 }} transition={{ duration: rv.t(0.6), delay: rv.t(0.1) * k + rv.t(0.3), ease }} /></span>
                <small>{note}</small>
              </li>
            ))}
          </ol>
        </li>
      </ol>
      <p className="heal-note rv" style={vars({ "--i": 4 })}>Every healed crop is mostly LaMa fill, so we checked that the retrained model did not just learn &ldquo;LaMa texture means no defect&rdquo;. We painted a clean patch into {int(D.n)} defect crops: it called {pct(D.called_no_defect.decoy)} of them no defect, against {pct(D.called_no_defect.original)} of the same crops untouched.</p>
    </div>
  );
}
