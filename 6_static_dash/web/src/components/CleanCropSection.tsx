import type React from "react";
import { motion } from "motion/react";
import { useLayoutEffect, useRef, useState } from "react";
import cc from "../data/cleancrop.json";
import healer from "../data/healer.json";
import { className, int, pct } from "../lib/format";
import { useReveal } from "../lib/useReveal";

/** scripts/clean_cropper_samples.py: one D1 photo through the clean cropper, its pick funnel, and d1_gc7's counts. */
const C = cc;
const DIR = "cleancrop/";
const vars = (v: Record<string, string | number>) => v as React.CSSProperties;
const ease = [0.22, 1, 0.36, 1] as const;
const ease3 = (t: number) => 1 - Math.pow(1 - t, 3);
const clamp = (v: number) => Math.max(0, Math.min(1, v));

const { w: W, h: H } = C.photo;
const PICK = C.candidates.findIndex((c) => c.picked);
const pickC = C.candidates[PICK];
/** Why each square is or isn't kept, in a few words, from its scores. */
const verdict = (c: (typeof C.candidates)[number]) =>
  c.picked ? `Clean ${pct(c.p_clean, 0)}` : c.p_skin < 0.2 ? "Background" : c.p_damage > 0.5 ? `Looks damaged ${pct(c.p_damage, 0)}` : `Clean ${pct(c.p_clean, 0)}`;

const STAGES: [string, React.ReactNode][] = [
  ["A labelled D1 photo", <>The red boxes are its labelled defects. Everything outside them is unlabelled, which does not make it clean: D1&rsquo;s labels miss some damage, and many photos show grass or sky.</>],
  ["Stay clear of every label", <>Each box is grown by half its size on every side, and no square may touch the shaded area. Boxes too small to become defect crops count too.</>],
  ["Try three squares", <>Up to three random squares per photo, each the size of this photo&rsquo;s own defect crops ({C.photo.sides.join(" or ")} px), none overlapping. Squares with black corners from D1&rsquo;s rotated copies are skipped.</>],
  ["Score each square", <>CLIP, a network pretrained on images and their captions, rates how much a square looks like aircraft skin and how much like a labelled defect. A small classifier trained on our hand labels turns that into p(clean). Here: grass, a loose wire, and plain panel.</>],
  ["It becomes a datapoint", <>The best square is resized to 224 × 224 and labelled <b>no defect</b>, in its photo&rsquo;s split. At most one per photo and ten per photo group, so no scene dominates.</>],
];
/** Where each stage reads best, as a fraction of the scroll through the scene; presentation mode stops here. */
const CC_AT = [0.04, 0.27, 0.5, 0.72, 0.96];

/** The clean cropper on one photo, then how the 3,000 were chosen, then d1_gc7 put together. */
export function CleanCropSection({ first, dimAt }: { first: number; dimAt(...steps: number[]): boolean }) {
  const steps = CC_AT.map((_, i) => first + i);
  return (
    <section className="cc-sec">
      <h2 className="data-h2">Finding clean squares in the photos</h2>
      <p className="how-sub cc-lede">Healing gives {int(healer.kept)} no-defect crops, all at the spot of a former defect. For skin that was never damaged, we cut squares from the parts of D1 photos nobody labelled. A hand check showed only half of those are clean, so every square is scored before it is kept.</p>
      <CleanScene steps={steps} dim={dimAt(...steps)} />
      <HowItPicks step={first + CC_AT.length} dim={dimAt(first + CC_AT.length)} />
    </section>
  );
}

function CleanScene({ steps, dim }: { steps: number[]; dim: boolean }) {
  const run = useRef<HTMLDivElement>(null), stage = useRef<HTMLDivElement>(null), frame = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState(0);

  useLayoutEffect(() => {
    const runEl = run.current!, st = stage.current!, fr = frame.current!;
    let raf = 0, lastPhase = -1;
    const frameFn = () => {
      raf = 0;
      const span = runEl.offsetHeight - st.offsetHeight;
      const p = clamp((st.getBoundingClientRect().top - runEl.getBoundingClientRect().top) / span);
      const keep = ease3(clamp((p - 0.13) / 0.12)), squares = clamp((p - 0.36) / 0.16);
      const score = ease3(clamp((p - 0.58) / 0.12)), lift = ease3(clamp((p - 0.8) / 0.14));
      fr.style.setProperty("--keep", String(keep));
      fr.style.setProperty("--score", String(score));
      fr.style.setProperty("--lift", String(lift));
      C.candidates.forEach((_, k) => fr.style.setProperty(`--sq${k}`, String(ease3(clamp(squares * 2.2 - k * 0.55)))));
      const ph = p < 0.2 ? 0 : p < 0.42 ? 1 : p < 0.64 ? 2 : p < 0.86 ? 3 : 4;
      if (ph !== lastPhase) { lastPhase = ph; setPhase(ph); }
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(frameFn); };
    frameFn();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, []);

  const at = (c: { left: number; top: number }) => ({ left: `${(c.left / W) * 100}%`, top: `${(c.top / H) * 100}%` });
  return (
    <div className="scene-run cc-run present-dim" data-dim={dim} ref={run} aria-label="How the clean cropper turns an unlabelled part of a photo into a no-defect datapoint">
      {CC_AT.map((a, i) => <i key={a} className="stage-mark" data-step={steps[i]} data-step-at={a} aria-hidden="true" />)}
      <div className="scene crop-stage" ref={stage}>
        <div className="crop-frame cc-frame" ref={frame} data-phase={phase}>
          <img className="cc-photo" src={DIR + "photo.jpg"} alt={`D1 photo ${C.photo.key}: a wing panel over grass, with a wire lying across it`} />
          <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
            <defs>
              <pattern id="cc-hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="10" height="10" className="cc-hatch-bg" /><line x1="0" y1="0" x2="0" y2="10" className="cc-hatch-line" />
              </pattern>
            </defs>
            {C.keep_out.map(([x1, y1, x2, y2], k) => <rect key={`k${k}`} className="cc-keep" x={x1} y={y1} width={x2 - x1} height={y2 - y1} />)}
            {C.boxes.map(([x1, y1, x2, y2], k) => <rect key={`b${k}`} className="cc-box" x={x1} y={y1} width={x2 - x1} height={y2 - y1} vectorEffect="non-scaling-stroke" />)}
            {C.candidates.map((c, k) => (
              <rect key={`c${k}`} className={`cc-sq${c.picked ? " is-pick" : ""}`} style={{ opacity: `var(--sq${k})` }} x={c.left} y={c.top} width={c.side} height={c.side} vectorEffect="non-scaling-stroke" />
            ))}
          </svg>
          {C.candidates.map((c, k) => (
            <span key={k} className={`cc-tag${c.picked ? " is-pick" : ""}`} style={{ left: at(c).left, top: at(c).top }}>{verdict(c)}</span>
          ))}
          <div className="cc-dim" aria-hidden="true" />
          <figure className="cc-point" style={vars({ "--x0": pickC.left / W, "--y0": pickC.top / H, "--s0": pickC.side / W })}>
            <img src={`${DIR}c${PICK}.jpg`} alt="The kept square at 224 by 224 pixels" />
            <figcaption><b>no defect</b><span>{C.photo.split} split · group {C.photo.group} · 224 × 224</span></figcaption>
          </figure>
        </div>
        <ol className="crop-steps" aria-live="polite">
          {STAGES.map(([head, text], i) => (
            <li key={head} className={phase === i ? "on" : undefined}><strong><span className="step-n">{i + 1}</span>{head}</strong><span><span>{text}</span></span></li>
          ))}
        </ol>
      </div>
    </div>
  );
}

const AUDIT: [string, string, string][] = [
  ["clean", "Clean surface", "var(--clear)"], ["background", "Background", "var(--unfamiliar)"],
  ["damage", "Damage nobody labelled", "var(--detected)"], ["other", "Blurry, dark or unclear", "var(--bar)"],
];

function HowItPicks({ step, dim }: { step: number; dim: boolean }) {
  const rv = useReveal<HTMLDivElement>(0.8);
  const f = C.funnel, share = f.audit_share as Record<string, number>, top = f.clean_in_top as Record<string, number>;
  const lift: [string, number][] = [["Top 10%", top["10%"]], ["Top 20%", top["20%"]], ["Top 30%", top["30%"]], ["At random", share.clean]];
  const funnel: [string, number, string][] = [
    ["Candidate squares", f.candidates, `from ${int(f.photos)} photos in ${f.groups} photo groups`],
    ["Picked", f.picked, `${f.per_bin} from each of ${f.bins} bands of detail, best p(clean) first`],
    ["Kept after review", f.kept, `from ${f.picked_groups} groups; a person checked every one and rejected none`],
  ];
  return (
    <div className="heal-how cc-how present-dim" data-step={step} data-dim={dim} ref={rv.ref} data-shown={rv.shown}>
      <h3 className="heal-h3 rv" style={vars({ "--i": 0 })}>How we picked {int(f.kept)}</h3>
      <ol className="heal-steps">
        <li className="rv" style={vars({ "--i": 1 })}>
          <strong><span className="step-n">1</span>Label a random sample</strong>
          <p>We labelled {f.audit} random candidates by hand, at most two per photo group. No single score sorted them well enough, so these labels train one.</p>
          <div className="cc-audit" role="img" aria-label={AUDIT.map(([k, label]) => `${label} ${pct(share[k], 0)}`).join(", ")}>
            {AUDIT.map(([k, , col], i) => (
              <motion.i key={k} style={{ background: col }} initial={false} animate={{ flexGrow: rv.shown ? share[k] : 0.0001 }}
                transition={{ duration: rv.t(0.6), delay: rv.t(0.08) * i + rv.t(0.2), ease }}>{share[k] >= 0.15 && <span className="fig">{pct(share[k], 0)}</span>}</motion.i>
            ))}
          </div>
          <ul className="cc-audit-key">{AUDIT.map(([k, label, col]) => <li key={k}><i style={{ background: col }} />{label} <span className="fig">{pct(share[k], 0)}</span></li>)}</ul>
        </li>
        <li className="rv" style={vars({ "--i": 2 })}>
          <strong><span className="step-n">2</span>Learn what clean looks like</strong>
          <p>A logistic regression on CLIP&rsquo;s features and scores, trained on those labels, ranks every candidate by p(clean). Checked on photo groups it never saw, its AUC is {f.auc.toFixed(2)}. Share of clean squares among its best-ranked:</p>
          <ol className="cc-lift">
            {lift.map(([label, v], i) => (
              <li key={label} className={i === lift.length - 1 ? "is-base" : undefined}>
                <span>{label}</span>
                <span className="heal-funnel-track"><motion.i initial={false} animate={{ scaleX: rv.shown ? v : 0 }} transition={{ duration: rv.t(0.6), delay: rv.t(0.08) * i + rv.t(0.3), ease }} /></span>
                <span className="fig">{pct(v, 0)}</span>
              </li>
            ))}
          </ol>
        </li>
        <li className="rv" style={vars({ "--i": 3 })}>
          <strong><span className="step-n">3</span>Match the detail, then review</strong>
          <p>Ranking alone picked mostly blank panels, which would teach the model &ldquo;featureless means no defect&rdquo;. So the picks are spread over {f.bins} bands of detail, from blank to busy, and keep rivets, seams and panel edges.</p>
          <ol className="heal-funnel">
            {funnel.map(([label, n, note], k) => (
              <li key={label} className={k === funnel.length - 1 ? "is-kept" : undefined}>
                <span className="heal-funnel-row"><span>{label}</span><span className="fig">{int(n)}</span></span>
                <span className="heal-funnel-track"><motion.i initial={false} animate={{ scaleX: rv.shown ? Math.max(n / f.candidates, 0.01) : 0 }} transition={{ duration: rv.t(0.6), delay: rv.t(0.1) * k + rv.t(0.3), ease }} /></span>
                <small>{note}</small>
              </li>
            ))}
          </ol>
        </li>
      </ol>
    </div>
  );
}

const SPLITS = ["train", "val", "test"] as const;
const ORDER = ["dent", "crack", "missing-head", "corrosion", "paint-peel-off", "scratch", "no-defect"];

/** d1_gc7: d1_grouped's six defect classes plus the no-defect crops from both modules, per class and split. */
export function Gc7Section({ step, dim }: { step: number; dim: boolean }) {
  const rv = useReveal<HTMLDivElement>(0.8);
  const g = C.gc7, cls = g.classes as Record<string, Record<string, number>>, nd = g.no_defect as Record<string, Record<string, number>>;
  const tot = (c: string) => SPLITS.reduce((a, s) => a + cls[c][s], 0);
  const max = Math.max(...ORDER.map(tot));
  const all = ORDER.reduce((a, c) => a + tot(c), 0);
  const ndTot = tot("no-defect"), healed = SPLITS.reduce((a, s) => a + nd.healed[s], 0), clean = SPLITS.reduce((a, s) => a + nd.clean_cropper[s], 0);
  return (
    <section className="gc7-sec present-dim" data-step={step} data-dim={dim} ref={rv.ref} data-shown={rv.shown}>
      <h2 className="data-h2 rv" style={vars({ "--i": 0 })}>Putting it together: d1_gc7</h2>
      <div className="gc7">
        <div>
          <p className="how-sub rv" style={vars({ "--i": 1 })}>The dataset the model trains and is tested on: d1_grouped&rsquo;s six defect classes, unchanged, plus a 7th class, <b>no defect</b>, made of {int(healed)} healed and {int(clean)} clean crops.</p>
          <ul className="gc7-facts rv" style={vars({ "--i": 2 })}>
            <li><span className="fig">{int(all)}</span> crops in 7 classes</li>
            <li>Every crop stays in its photo&rsquo;s d1_grouped split, so no test photo reaches training in any form.</li>
            <li>Taken whole, the no-defect crops came out about 59 / 21 / 20, since validation and test have many small groups. So they keep the defect classes&rsquo; 70 / 15 / 15; the {int(g.extra)} left over are a spare check set the model never trains on.</li>
          </ul>
        </div>
        <div className="rv" style={vars({ "--i": 2 })}>
          <ol className="gc7-bars" aria-label="Crops per class and split">
            {ORDER.map((c, i) => (
              <li key={c} className={c === "no-defect" ? "is-new" : undefined}>
                <span className="gc7-name">{className(c)}</span>
                <span className="gc7-track">
                  {SPLITS.map((s, k) => (
                    <motion.i key={s} className={`gc7-${s}`} initial={false} animate={{ width: rv.shown ? `${(cls[c][s] / max) * 100}%` : "0%" }}
                      transition={{ duration: rv.t(0.6), delay: rv.t(0.05) * i + rv.t(0.12) * k, ease }} />
                  ))}
                </span>
                <span className="fig">{int(tot(c))}</span>
              </li>
            ))}
          </ol>
          <p className="chart-key gc7-key"><span><i className="gc7-train" /> Train</span><span><i className="gc7-val" /> Validation</span><span><i className="gc7-test" /> Test</span>
            <span className="gc7-src">No defect: {int(healed)} healed + {int(clean)} clean = {int(ndTot)}</span></p>
        </div>
      </div>
    </section>
  );
}
