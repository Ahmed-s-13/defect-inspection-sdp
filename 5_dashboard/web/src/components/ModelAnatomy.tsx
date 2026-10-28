import type React from "react";
import { animate, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import anatomy from "../data/anatomy.json";
import { className, int, pct } from "../lib/format";
import { useReveal, type Reveal } from "../lib/useReveal";
import { usePresent } from "../state/present";
import { RuleDraw } from "./RuleDraw";
import { STATE_WORDS } from "./StateBadge";

/** scripts/model_anatomy.py: one test crop followed through enhanced_res, every number from the real model. */
const A = anatomy;
const DIR = "/anatomy/";
const vars = (v: Record<string, string | number>) => v as React.CSSProperties;
const ease = [0.22, 1, 0.36, 1] as const;
const top = (p: number[]) => p.indexOf(Math.max(...p));
const CROP_CLS = className(A.crop.cls).toLowerCase();
/** 0.5 -> "Half", 0.75 -> "Three quarters", 0.25 -> "A quarter of" */
const share = (v: number) => ({ 0.5: "Half", 0.75: "Three quarters of", 0.25: "A quarter of" } as Record<number, string>)[v] ?? `${Math.round(v * 100)}% of`;

/** The section's presentation steps: one per card. */
export const ANATOMY_STEPS = 7;

export function ModelAnatomy({ first, dimAt }: { first: number; dimAt(step: number): boolean }) {
  const head = useReveal<HTMLDivElement>(0.9);
  const l3 = A.layers[2].channels, l4 = A.layers[3].channels;
  const sameGroup = A.neighbours.filter((n) => n.group === A.neighbours[0].group).length;
  const nTrue = A.neighbours.filter((n) => n.cls === A.crop.cls).length;
  const seven = A.ladder.some((r) => "seven" in r);
  const cards: Card[] = [
    ["Look at the crop 8 ways",
      <>D1 is full of flipped and rotated copies, so the answer should not depend on which way a photo was taken. Each crop is turned 4 ways and mirrored, and everything after this is averaged over the 8 views.</>,
      "1 crop", "8 views", (rv) => <Views rv={rv} />],
    ["A ResNet-18 finds the patterns",
      <>It starts from weights pretrained on ImageNet (1.2 million everyday photos) and was retrained on our training crops, no-defect crops included, with each photo group and each class drawn equally often. Every stage halves the picture and tracks more patterns: edges, then textures, then parts of defects.</>,
      "224 × 224 px", `${A.layers[3].size} × ${A.layers[3].size} × ${l4}`, (rv) => <Layers rv={rv} />],
    ["Keep 768 numbers",
      <>Each of layer 3's {l3} patterns and layer 4's {l4} is averaged over the whole square, then over the 8 views. Layer 3 keeps mid-level texture that layer 4 summarises away.</>,
      "8 views", `${A.d_in} numbers`, (rv) => <Features rv={rv} />],
    ["Five small networks vote",
      <>Five copies of one MLP ({[A.d_in, ...A.hidden, A.classes.length].map(int).join(" → ")}), trained on these numbers from different random starts. Their probabilities are averaged, which steadies the answer.</>,
      `${A.d_in} numbers`, `${pct(A.mlp[top(A.mlp)])} ${CROP_CLS}`, (rv) => <Heads rv={rv} />],
    [`Ask the ${A.k} nearest training crops`,
      <>At the same time it looks up the {A.k} training crops whose numbers are closest, out of {int(A.bank)}, and lets them vote, closer ones counting more. Here {nTrue === A.k ? `all ${A.k} are ${CROP_CLS}s` : `${nTrue} of ${A.k} are ${CROP_CLS}s`}{sameGroup > 1 ? ` (${sameGroup} are copies of one training photo)` : ""}. The same distance tells architecture B when a crop is unlike anything in training.</>,
      `${int(A.bank)} training crops`, `${A.k} neighbours`, (rv) => <Neighbours rv={rv} />],
    [A.tau > 0 ? "Blend, correct, then calibrate" : "Blend, then calibrate",
      <>{share(1 - A.alpha)} the networks' vote, {share(A.alpha).toLowerCase()} the neighbours'.{A.tau > 0 ? <> Each class is then divided by the square root of its share of training, so the big classes (dent, crack) don't win by default. Both weights were picked on validation.</> : null} Architecture B then softens the result with one temperature (T = {A.temperature.toFixed(2)}) so the percentage is honest, and decides what to show, as the next sections explain.</>,
      "two votes", `${STATE_WORDS[A.state as "detected"]}: ${A.answer.map(className).join(" or ")}`, (rv) => <Blend rv={rv} />],
    ["What each part added",
      <>Macro-F1 after each change, every class counting equally. Each change was kept because validation went up; the 8 views gave the biggest jump. On test one step cost a little, which we accepted rather than choose on test.{seven ? " The last row is today's model: the same recipe with the no-defect class, scored over all 7 classes." : ""}</>,
      "plain model", "enhanced_res", () => <Ladder />],
  ];
  return (
    <div className="anatomy">
      <div className="panel" ref={head.ref} data-shown={head.shown}>
        <RuleDraw />
        <h2 className="panel-title rv" style={vars({ "--i": 0 })}>How it works</h2>
        <p className="panel-sub rv" style={vars({ "--i": 1 })}>One test crop, a {CROP_CLS} from a photo group the model never trained on, followed through enhanced_res. Every picture and number below is the real model on this crop.</p>
      </div>
      <Deck cards={cards} first={first} dimAt={dimAt} />
    </div>
  );
}

type Card = [string, ReactNode, string, string, (rv: Reveal) => ReactNode];
const FLY = 0.62; // seconds for one card to go round the pile

/** The cards as a pile: the front card goes to the back to show the next one. Swipe or drag sideways, use the
 *  arrow keys or the buttons; in presentation mode each card is a step. */
function Deck({ cards, first, dimAt }: { cards: Card[]; first: number; dimAt(step: number): boolean }) {
  const n = cards.length;
  const [active, setActive] = useState(0);
  const rv = useReveal<HTMLDivElement>(0.8);
  const reduce = useReducedMotion();
  const present = usePresent();
  const els = useRef<(HTMLElement | null)[]>([]);
  const stage = useRef<HTMLDivElement>(null);
  const shown = useRef(0), busy = useRef(false), dragX = useRef(0);

  // where each card rests: the front one flat, the next three stepping down and back, the rest hidden
  const rest = (depth: number) => ({ x: 0, rotate: 0, y: Math.min(depth, 3) * 14, scale: 1 - Math.min(depth, 3) * 0.035, opacity: depth > 3 ? 0 : 1 });
  const place = (to: number, mover: number | null, forward: boolean) => {
    els.current.forEach((el, k) => {
      if (!el) return;
      const depth = (k - to + n) % n, r = rest(depth);
      el.style.setProperty("--depth", String(Math.min(depth, 3))); // cards further back are a shade darker
      if (k !== mover || reduce) {
        el.style.zIndex = String(n - depth);
        animate(el, r, reduce ? { duration: 0 } : { duration: 0.45, ease, delay: forward ? 0.12 : 0 });
        return;
      }
      // the mover swings out to the side, then tucks behind the pile (forward) or comes out of it to the front (back)
      const out = -(stage.current?.offsetWidth ?? 800) * 0.62;
      el.style.zIndex = String(forward ? n + 1 : 0);
      window.setTimeout(() => { el.style.zIndex = String(n - depth); }, FLY * 500);
      // going back it lands on the last visible place in the pile, then hides behind the cards in front of it
      const land = forward ? rest(3) : r;
      animate(el, {
        x: [forward ? dragX.current : 0, out, 0], rotate: [0, forward ? -6 : 6, 0], y: [forward ? 0 : land.y, 0, land.y],
        scale: [forward ? 1 : land.scale, 0.97, land.scale], opacity: forward ? 1 : [0, 1, 1],
      }, { duration: FLY, ease: "easeInOut", times: [0, 0.5, 1] }).then(() => { if (forward) animate(el, r, { duration: 0 }); });
    });
  };
  const go = (target: number) => {
    const from = shown.current, to = ((target % n) + n) % n;
    if (to === from || busy.current) return;
    const forward = to === (from + 1) % n || (to > from && !(from === 0 && to === n - 1));
    const oneStep = to === (from + 1) % n || from === (to + 1) % n;
    busy.current = true;
    window.setTimeout(() => { busy.current = false; }, reduce ? 0 : FLY * 600);
    place(to, oneStep ? (forward ? from : to) : null, forward); // a jump of several cards moves straight there
    dragX.current = 0;
    shown.current = to;
    setActive(to);
  };
  useEffect(() => { place(0, null, true); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  // presentation mode: the arrow keys walk the cards while this section is the current step
  useEffect(() => {
    const k = present.step - first;
    if (present.on && k >= 0 && k < n) { busy.current = false; go(k); }
  }, [present.on, present.step]); // eslint-disable-line react-hooks/exhaustive-deps

  // a sideways trackpad swipe turns one card
  useEffect(() => {
    const el = stage.current!;
    let acc = 0, lock = 0;
    const onWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;
      e.preventDefault();
      if (performance.now() < lock) return;
      acc += e.deltaX;
      if (Math.abs(acc) > 40) { go(shown.current + (acc > 0 ? 1 : -1)); acc = 0; lock = performance.now() + 700; }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // drag the front card; let go past a threshold to turn it
  const start = useRef<{ x: number; id: number } | null>(null);
  const front = () => els.current[shown.current];
  const onDown = (e: PointerEvent) => {
    if (e.button !== 0 || busy.current) return;
    start.current = { x: e.clientX, id: e.pointerId };
  };
  const onMove = (e: PointerEvent) => {
    const s = start.current;
    if (!s) return;
    const dx = e.clientX - s.x;
    if (Math.abs(dx) > 6 && !(e.currentTarget as Element).hasPointerCapture(s.id)) (e.currentTarget as Element).setPointerCapture(s.id);
    dragX.current = Math.min(dx, 60 + dx * 0.15);
    const el = front();
    if (el) animate(el, { x: dragX.current, rotate: dragX.current / 60 }, { duration: 0 });
  };
  const onUp = () => {
    if (!start.current) return;
    start.current = null;
    if (dragX.current < -70) go(shown.current + 1);
    else if (dragX.current > 50) go(shown.current - 1);
    else { const el = front(); if (el) animate(el, { x: 0, rotate: 0 }, { duration: 0.35, ease }); dragX.current = 0; }
  };
  const onKey = (e: KeyboardEvent) => {
    if (present.on) return; // the presentation's own arrow keys turn the cards
    if (e.key === "ArrowRight") { e.preventDefault(); go(shown.current + 1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); go(shown.current - 1); }
  };

  const steps = cards.map((_, i) => first + i);
  return (
    <div className="deck present-dim" data-dim={steps.every(dimAt)} ref={rv.ref}>
      {steps.map((s) => <i key={s} className="deck-mark" data-step={s} aria-hidden="true" />)}
      <div className="deck-stage" ref={stage} tabIndex={0} role="group" aria-roledescription="carousel" aria-label="How enhanced_res works, step by step"
        onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} onKeyDown={onKey}>
        {cards.map(([title, text, from, to, visual], i) => {
          const cardRv: Reveal = { shown: rv.shown && i === active, t: rv.t };
          return (
            <article key={title} ref={(el) => { els.current[i] = el; }} className={`deck-card${i === active ? " is-front" : ""}`}
              data-shown={cardRv.shown} aria-hidden={i !== active} aria-roledescription="slide" aria-label={`${i + 1} of ${n}: ${title}`}>
              <div>
                <p className="deck-count rv" style={vars({ "--i": 0 })}><span className="step-n">{i + 1}</span> of {n}</p>
                <h3 className="an-title rv" style={vars({ "--i": 0 })}>{title}</h3>
                <p className="an-text rv" style={vars({ "--i": 1 })}>{text}</p>
                <p className="an-io rv" style={vars({ "--i": 2 })}>{from} → <b>{to}</b></p>
              </div>
              <div className="an-visual">{visual(cardRv)}</div>
            </article>
          );
        })}
      </div>
      <div className="deck-nav">
        <button className="deck-btn" onClick={() => go(active - 1)} aria-label="Previous card"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7" /></svg></button>
        <div className="deck-dots" role="tablist" aria-label="Cards">
          {cards.map(([title], i) => (
            <button key={title} role="tab" aria-selected={i === active} aria-label={`${i + 1}: ${title}`} className="deck-dot" onClick={() => go(i)} />
          ))}
        </div>
        <button className="deck-btn" onClick={() => go(active + 1)} aria-label="Next card"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7" /></svg></button>
        <span className="deck-hint">Swipe, drag or use ← → to turn the cards</span>
      </div>
    </div>
  );
}

// torch.rot90 turns counter-clockwise; views 4-7 are the same turns, then mirrored
const VIEW_T = Array.from({ length: 8 }, (_, k) => `${k >= 4 ? "scaleX(-1) " : ""}rotate(${-90 * (k % 4)}deg)`);

function Views({ rv }: { rv: Reveal }) {
  return (
    <div className="an-views">
      {VIEW_T.map((t, k) => (
        <motion.span key={k} className="an-view" initial={false} animate={{ opacity: rv.shown ? 1 : 0, scale: rv.shown ? 1 : 0.85 }}
          transition={{ duration: rv.t(0.4), delay: rv.t(0.05) * k, ease }}>
          <img src={DIR + "crop.jpg"} alt={k ? "" : `The test crop: a ${CROP_CLS}`} style={{ transform: t }} />
          <i>{k < 4 ? `${90 * k}°` : `${90 * (k - 4)}° mirrored`}</i>
        </motion.span>
      ))}
    </div>
  );
}

function Layers({ rv }: { rv: Reveal }) {
  const tiles = [{ src: "crop.jpg", label: "crop", sub: "224 × 224 × 3", s: 1 },
    ...A.layers.map((l, i) => ({ src: `${l.name}.png`, label: l.name.replace("layer", "layer "), sub: `${l.size} × ${l.size} × ${l.channels}`, s: 0.88 - i * 0.12 }))];
  return (
    <div className="an-layers">
      {tiles.map((t, i) => (
        <motion.figure key={t.src} className={`an-layer${i >= 3 ? " is-kept" : ""}`} initial={false}
          animate={{ opacity: rv.shown ? 1 : 0, x: rv.shown ? 0 : -12 }} transition={{ duration: rv.t(0.45), delay: rv.t(0.12) * i, ease }}>
          <img src={DIR + t.src} alt={i ? `${t.label} activity: darker teal is stronger` : ""} style={{ width: `${t.s * 100}%` }} />
          <figcaption><b>{t.label}</b><span>{t.sub}</span></figcaption>
        </motion.figure>
      ))}
      <p className="an-note">Teal shows where each stage reacts most. By layer 4 it is the {CROP_CLS}. Layers 3 and 4 feed the next step.</p>
    </div>
  );
}

function Features({ rv }: { rv: Reveal }) {
  const l3 = A.layers[2].channels, BIN = 4;
  const bars = Array.from({ length: A.features.length / BIN }, (_, i) => A.features.slice(i * BIN, i * BIN + BIN).reduce((a, b) => a + b, 0) / BIN);
  const max = Math.max(...bars);
  return (
    <div className="an-feat">
      <div className="an-feat-bars" role="img" aria-label={`${A.d_in} feature values: ${l3} from layer 3, ${A.d_in - l3} from layer 4`}>
        {bars.map((v, i) => (
          <motion.i key={i} className={i * BIN < l3 ? "l3" : "l4"} initial={false}
            animate={{ scaleY: rv.shown ? Math.max(0.03, Math.sqrt(v / max)) : 0 }} transition={{ duration: rv.t(0.5), delay: rv.t(0.004) * i, ease }} />
        ))}
      </div>
      <div className="an-feat-key" style={vars({ "--l3": l3 / A.d_in })}><span>layer 3 · {l3}</span><span>layer 4 · {A.d_in - l3}</span></div>
      <p className="an-note">Each bar is the average of 4 neighbouring numbers; height is the square root, so small values stay visible.</p>
    </div>
  );
}

function Heads({ rv }: { rv: Reveal }) {
  const c = top(A.mlp);
  const cols = [A.d_in, ...A.hidden, A.classes.length];
  return (
    <div className="an-heads">
      <svg className="an-net" viewBox="0 0 300 120" aria-hidden="true">
        {cols.map((n, j) => {
          const k = Math.max(3, Math.round(Math.log2(n) * 1.2)), x = 20 + j * 65;
          return Array.from({ length: k }, (_, i) => {
            const y = 60 + (i - (k - 1) / 2) * (100 / 12);
            return (
              <g key={`${j}-${i}`}>
                {j < cols.length - 1 && <line x1={x} y1={y} x2={x + 65} y2={60} className="an-edge" />}
                <circle cx={x} cy={y} r={j === cols.length - 1 && i === c ? 4 : 2.6} className={j === cols.length - 1 && i === c ? "an-node is-top" : "an-node"} />
              </g>
            );
          });
        })}
        {cols.map((n, j) => <text key={j} x={20 + j * 65} y="118" textAnchor="middle" className="an-net-n">{int(n)}</text>)}
      </svg>
      <ol className="an-votes">
        {A.heads.map((p, i) => (
          <li key={i}><span>Network {i + 1}</span>
            <span className="an-track"><motion.i initial={false} animate={{ scaleX: rv.shown ? p[c] : 0 }} transition={{ duration: rv.t(0.6), delay: rv.t(0.08) * i, ease }} /></span>
            <span className="fig">{pct(p[c])}</span></li>
        ))}
        <li className="is-mean"><span>Average</span>
          <span className="an-track"><motion.i initial={false} animate={{ scaleX: rv.shown ? A.mlp[c] : 0 }} transition={{ duration: rv.t(0.6), delay: rv.t(0.45), ease }} /></span>
          <span className="fig">{pct(A.mlp[c])}</span></li>
      </ol>
      <p className="an-note">Probability of {CROP_CLS} from each network.</p>
    </div>
  );
}

function Neighbours({ rv }: { rv: Reveal }) {
  return (
    <div className="an-nn">
      <figure className="an-query"><img src={DIR + "crop.jpg"} alt="" /><figcaption>this crop</figcaption></figure>
      <ol>
        {A.neighbours.map((n, i) => (
          <motion.li key={i} initial={false} animate={{ opacity: rv.shown ? 1 : 0, y: rv.shown ? 0 : 10 }} transition={{ duration: rv.t(0.4), delay: rv.t(0.08) * i, ease }}>
            <img src={`${DIR}nn${i}.jpg`} alt={`Training crop: ${className(n.cls)}`} loading="lazy" />
            <span><b>{className(n.cls)}</b><span className="fig">{n.similarity.toFixed(2)}</span></span>
          </motion.li>
        ))}
      </ol>
      <p className="an-note">Under each: its class and how similar its numbers are (cosine, 1 = identical).</p>
    </div>
  );
}

function Blend({ rv }: { rv: Reveal }) {
  const c = top(A.blend);
  const rows: [string, number, string][] = [
    ["Networks", A.mlp[c], `× ${(1 - A.alpha).toFixed(2)}`], ["Neighbours", A.knn[c], `× ${A.alpha.toFixed(2)}`],
    ["Blend", A.blend[c], ""], ...(A.tau > 0 ? [["Corrected", A.corrected[c], A.tau === 0.5 ? "÷ √share" : `÷ share^${A.tau}`] as [string, number, string]] : []),
    ["Calibrated", A.tempered[c], `T = ${A.temperature.toFixed(2)}`],
  ];
  return (
    <div className="an-blend">
      <ol className="an-votes">
        {rows.map(([label, v, note], i) => (
          <li key={label} className={i === rows.length - 1 ? "is-mean" : undefined}><span>{label} <small>{note}</small></span>
            <span className="an-track"><motion.i initial={false} animate={{ scaleX: rv.shown ? v : 0 }} transition={{ duration: rv.t(0.6), delay: rv.t(0.15) * i, ease }} /></span>
            <span className="fig">{pct(v)}</span></li>
        ))}
      </ol>
      <p className="an-note">Probability of {CROP_CLS}. The other {A.classes.length - 1} classes share the rest.</p>
    </div>
  );
}

/** Macro-F1 after each change, on validation (where every choice was made) and on test. */
function Ladder() {
  const lo = 0.64, hi = 0.76, x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <>
      <ol className="ladder">
        {A.ladder.map((r) => (
          <li key={r.label} className={"seven" in r ? "is-seven" : undefined}>
            <span className="ladder-label">{r.label}</span>
            <span className="ladder-track">
              <span className="ladder-link" style={{ left: x(Math.min(r.val, r.test)), width: `calc(${x(Math.max(r.val, r.test))} - ${x(Math.min(r.val, r.test))})` }} />
              <span className="ladder-dot is-val" style={{ left: x(r.val) }} />
              <span className="ladder-dot is-test" style={{ left: x(r.test) }} />
            </span>
            <span className="fig">{r.test.toFixed(3)}</span>
          </li>
        ))}
      </ol>
      <p className="chart-key"><span className="key-dot key-before" /> Validation <span className="key-dot key-now" /> Test (figure) · axis {lo.toFixed(2)}–{hi.toFixed(2)}</p>
    </>
  );
}
