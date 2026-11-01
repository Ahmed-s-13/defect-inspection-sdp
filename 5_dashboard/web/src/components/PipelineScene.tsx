import type React from "react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Ranked, Report, StateId } from "../api/client";
import pipe from "../data/pipeline.json";
import { className, int, pct } from "../lib/format";
import { useReveal } from "../lib/useReveal";
import { AonScatter, ClassCoverage, KnnHist } from "./charts/PipelineCharts";
import { RuleDraw } from "./RuleDraw";
import { SoftmaxBars } from "./SoftmaxBars";
import { StateGlyph } from "./StateBadge";
import "../styles/pipeline.css";

/** scripts/pipeline_scene.py: five test crops through architecture B, one per visor outcome, with the numbers each
 *  stage works on, the alpha grid and the data behind the stage details. */
export type Outcome = "show" | "ambiguous" | "possible" | "nothing" | "reaim";
export interface PipeCrop {
  outcome: Outcome; image: string; truth: string; raw: number[]; tempered: number[];
  knn_dist: number; aon_score: number; p_knn: number; p_aon: number; p_ood: number; set: string[]; outcomes: Record<string, string>;
  about?: string; credit?: { author: string; url: string; license: string; license_url: string; source: string };
}
/** The re-aim example is not a D1 crop but a non-aircraft photo (truth "off-target"), shown with its credit. */
export const isOff = (c: PipeCrop) => c.truth === "off-target";
export const PIPE = pipe as unknown as {
  made: string; classes: string[]; T: number; served: { ood: number; cp: number }; ood_alphas: number[]; cp_alphas: number[];
  test_half: { n: number; families: number }; crops: PipeCrop[];
  grid: Record<string, { q: number; coverage: number; family_coverage: number; set_size: number; mix: Record<string, number>;
    silent_miss: number; clean_silent: number; reaim: number; wrong_no_warning: number; attention: number; class_coverage: Record<string, number> }>;
  knn: { edges: number[]; counts: number[]; n: number; cut: Record<string, number> };
  aon: { skin: [number, number][]; off: [number, number][]; crops: [number, number][]; cut: Record<string, number> };
};
export const cellKey = (ood: number, cp: number) => `${ood.toFixed(2)}|${cp.toFixed(2)}`;

/** The visor's buckets, in the order they are shown, with the dashboard's state for each. */
export const BUCKETS: [Outcome | "unclear", StateId, string][] = [
  ["show", "detected", "Defect detected"], ["ambiguous", "ambiguous", "Ambiguous"], ["possible", "ambiguous", "Possible: check"],
  ["nothing", "clear", "Nothing shown"], ["reaim", "cannot_assess", "Re-aim"], ["unclear", "unclear", "Unclear"],
];
export const bucketState = (o: string) => (BUCKETS.find(([k]) => k === o)?.[1] ?? "unclear");
export const bucketLabel = (o: string) => (BUCKETS.find(([k]) => k === o)?.[2] ?? o);

export const ABOUT: Record<Outcome, string> = {
  show: "a crack through a hole", ambiguous: "peeling paint, or corrosion?", possible: "a faint dent",
  nothing: "plain, undamaged skin", reaim: "a glove, not aircraft skin",
};

type Stage = "crop" | "model" | "ood" | "temp" | "set" | "visor";
const NODES: [Stage, string, string][] = [
  ["crop", "Test crop", "224 px, never trained on"], ["model", "enhanced_res gc7", "8 views · 7 classes"],
  ["ood", "Unfamiliar-input check", "kNN · aircraft or not"], ["temp", "Temperature", `T = ${PIPE.T.toFixed(2)}`],
  ["set", "Conformal set", "classes past the cut-off"], ["visor", "Visor", "what the inspector sees"],
];
// where each stage starts within one crop's stretch of the scroll; a re-aimed crop skips temperature and the set
const PATH: Record<"pass" | "stop", [Stage, number][]> = {
  pass: [["crop", 0], ["model", 0.1], ["ood", 0.27], ["temp", 0.45], ["set", 0.6], ["visor", 0.78]],
  stop: [["crop", 0], ["model", 0.1], ["ood", 0.27], ["visor", 0.58]],
};
const DROP_AT = 0.52; // within a crop's visor stage: when it leaves the Visor for its bucket
const INTRO = 0.05, SEG = 0.17, OUTRO = INTRO + PIPE.crops.length * SEG;
/** Presentation stops: the start, each crop at its conformal set (or its re-aim), and the five crops in their buckets. */
export const PIPE_AT = [0.02, ...PIPE.crops.map((c, i) => INTRO + (i + (c.outcome === "reaim" ? 0.68 : 0.7)) * SEG), 0.96];

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const smooth = (t: number) => t * t * (3 - 2 * t);
const ranked = (p: number[]): Ranked[] => PIPE.classes.map((cls, i) => ({ cls, p: p[i] })).sort((a, b) => b.p - a.p);
const pathOf = (c: PipeCrop) => PATH[c.outcome === "reaim" ? "stop" : "pass"];

/** Where the scroll is: before the first crop (-1), on crop i at one of its stages, or after the last (5). */
function where(p: number): { crop: number; stage: Stage; t: number; local: number } {
  if (p < INTRO) return { crop: -1, stage: "crop", t: 0, local: 0 };
  if (p >= OUTRO) return { crop: PIPE.crops.length, stage: "visor", t: 1, local: 1 };
  const crop = Math.floor((p - INTRO) / SEG), local = (p - INTRO) / SEG - crop;
  const path = pathOf(PIPE.crops[crop]);
  let k = path.length - 1;
  while (k > 0 && local < path[k][1]) k--;
  const end = k + 1 < path.length ? path[k + 1][1] : 1;
  return { crop, stage: path[k][0], t: (local - path[k][1]) / (end - path[k][1]), local };
}

const P_LO = 0.005; // the p-value lines run on a log scale from here to 1
const px = (p: number) => (Math.log(Math.max(p, P_LO)) - Math.log(P_LO)) / -Math.log(P_LO);
const fmtP = (p: number) => (p >= 0.995 ? "1.00" : p >= 0.1 ? p.toFixed(2) : p.toFixed(3));

/** Five real test crops walk architecture B as you scroll: each stage lights up as a crop reaches it, says what it
 *  does and shows what it does to this crop; each crop lands in its visor bucket, which keeps it. The unfamiliar-input
 *  check and the set open a chart of what they compare against. */
export function PipelineScene({ rep, first, dimAt }: { rep: Report; first: number; dimAt(step: number): boolean }) {
  const run = useRef<HTMLDivElement>(null), stage = useRef<HTMLDivElement>(null), thumb = useRef<HTMLDivElement>(null);
  const nodes = useRef<Partial<Record<Stage, HTMLElement | null>>>({});
  const slots = useRef<Partial<Record<string, HTMLElement | null>>>({});
  const [pos, setPos] = useState(() => where(0));
  const [detail, setDetail] = useState<"knn" | "aon" | "set" | null>(null);
  const rv = useReveal<HTMLDivElement>(0.9);
  const a = rep.arch_a, served = a.by_alpha[a.alpha.toFixed(2)];
  const cell = PIPE.grid[cellKey(PIPE.served.ood, PIPE.served.cp)];
  const cut = 1 - cell.q;

  useLayoutEffect(() => {
    const runEl = run.current!, st = stage.current!, th = thumb.current!;
    const bar = document.querySelector<HTMLElement>(".topbar")?.offsetHeight ?? 56;
    st.style.setProperty("--stick", `${bar}px`);
    let raf = 0, last = "";
    const centre = (el: HTMLElement | null | undefined, box: DOMRect): [number, number] => {
      if (!el) return [0, 0];
      const r = el.getBoundingClientRect();
      return [r.left - box.left + r.width / 2, r.top - box.top + r.height / 2];
    };
    const frame = () => {
      raf = 0;
      const p = clamp((st.getBoundingClientRect().top - runEl.getBoundingClientRect().top) / (runEl.offsetHeight - st.offsetHeight));
      const w = where(p), box = st.getBoundingClientRect();
      // the travelling crop: over the stage it is at, gliding from the last one; down into its bucket at the visor
      if (w.crop >= 0 && w.crop < PIPE.crops.length) {
        const c = PIPE.crops[w.crop], path = pathOf(c);
        const k = path.findIndex(([s]) => s === w.stage);
        const lane = (s: Stage) => { const [x] = centre(nodes.current[s], box); const r = nodes.current.model?.getBoundingClientRect(); return [x, (r ? r.top - box.top : 0) - th.offsetHeight / 2 - 8] as [number, number]; };
        const from = k === 0 ? [lane("crop")[0] - 40, lane("crop")[1]] : lane(path[k - 1][0]);
        // at the visor: first along the lane to the Visor stage, a pause, then down into its bucket
        const m = smooth(clamp(w.t / (w.stage === "visor" ? 0.35 : 0.4)));
        const to = lane(w.stage);
        let x = from[0] + (to[0] - from[0]) * m, y = from[1] + (to[1] - from[1]) * m, drop = 0;
        if (w.stage === "visor") {
          drop = smooth(clamp((w.t - DROP_AT) / 0.3));
          const slot = centre(slots.current[c.outcome], box);
          x += (slot[0] - x) * drop; y += (slot[1] - y) * drop;
        }
        const landed = w.stage === "visor" && drop >= 1;
        th.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%) scale(${1 - 0.3 * drop})`;
        th.style.opacity = String(landed ? 0 : clamp(w.local / 0.04));
      } else th.style.opacity = "0";
      const key = `${w.crop}:${w.stage}:${w.stage === "visor" && w.t >= DROP_AT + 0.3}`; // landing is a change too
      if (key !== last) { last = key; setPos(w); }
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(frame); };
    frame();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, []);

  // a detail closes when the crop moves on, on Escape, or on a click outside it
  useEffect(() => { setDetail(null); }, [pos.crop, pos.stage]);
  useEffect(() => {
    if (!detail) return;
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setDetail(null); };
    const down = (e: PointerEvent) => { if (!(e.target as Element).closest(".pl-detail, .pl-chip")) setDetail(null); };
    window.addEventListener("keydown", key);
    window.addEventListener("pointerdown", down);
    return () => { window.removeEventListener("keydown", key); window.removeEventListener("pointerdown", down); };
  }, [detail]);

  const done = pos.crop >= PIPE.crops.length, c = pos.crop >= 0 && !done ? PIPE.crops[pos.crop] : null;
  const path = c ? pathOf(c) : [];
  const k = path.findIndex(([s]) => s === pos.stage);
  const nodeState = (s: Stage) => {
    if (!c) return done ? "is-rest" : "";
    const i = path.findIndex(([x]) => x === s);
    return i < 0 ? "is-skipped" : i < k ? "is-lit" : i === k ? "is-now" : "";
  };
  const landed = (i: number) => i < pos.crop || (i === pos.crop && pos.stage === "visor" && pos.t >= DROP_AT + 0.3);
  const share: Record<string, number> = {
    show: served.actions["show class"], ambiguous: served.actions.ambiguous, possible: served.actions["possible defect"] ?? 0,
    nothing: served.actions["nothing shown"] ?? 0, reaim: served.actions["re-aim"],
  };
  const dim = PIPE_AT.every((_, i) => dimAt(first + i));
  const bars = c && (pos.stage === "model" || pos.stage === "temp" || pos.stage === "set" || pos.stage === "visor");
  const kept = c && (pos.stage === "set" || pos.stage === "visor") && c.outcome !== "reaim";

  return (
    <div className="panel pl-sec" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <div className="scene-run pl-run present-dim" data-dim={dim} ref={run} aria-label="Five test crops pass through architecture B's stages and land in the visor's buckets">
        {PIPE_AT.map((at, i) => <i key={at} className="stage-mark" data-step={first + i} data-step-at={at} aria-hidden="true" />)}
        <div className="scene pl-stage" ref={stage}>
          <div className="pl-head">
            <h2 className="panel-title">Follow a crop through architecture B</h2>
            <p className="panel-sub">Four test crops and one photo that isn't aircraft, one at a time: each stage lights up as a crop reaches it, and each crop lands in its visor bucket.</p>
          </div>
          <ol className="pl-row" aria-label="Architecture B's stages">
            {NODES.map(([s, title, sub]) => (
              <li key={s} ref={(el) => { nodes.current[s] = el; }} className={`pl-node ${nodeState(s)}`}>
                <strong>{title}</strong>
                <span>{sub}</span>
                {s === "ood" && (
                  <span className="pl-chips">
                    {(["knn", "aon"] as const).map((d) => (
                      <button key={d} type="button" className="pl-chip" aria-expanded={detail === d} disabled={!c || k < 2}
                        onClick={() => setDetail(detail === d ? null : d)}>{d === "knn" ? "kNN distance" : "Aircraft or not"}</button>
                    ))}
                  </span>
                )}
                {s === "set" && (
                  <span className="pl-chips">
                    <button type="button" className="pl-chip" aria-expanded={detail === "set"} disabled={!c || k < 4 || c.outcome === "reaim"}
                      onClick={() => setDetail(detail === "set" ? null : "set")}>Per class</button>
                  </span>
                )}
              </li>
            ))}
          </ol>

          <div className="pl-body">
            <div className="pl-note" key={`${pos.crop}:${pos.stage}`} aria-live="polite">
              <Note pos={pos} c={c} rep={rep} cut={cut} />
            </div>
            <div className="pl-visual">
              <figure className={`pl-layer pl-crop${c && pos.stage === "crop" ? " on" : ""}`}>
                {c && <img src={c.image} alt={`Test crop: ${ABOUT[c.outcome]}`} />}
                {c && (isOff(c) && c.credit
                  ? <figcaption>Not aircraft skin · photo <a href={c.credit.url} target="_blank" rel="noreferrer">{c.credit.author}</a>, <a href={c.credit.license_url} target="_blank" rel="noreferrer">{c.credit.license}</a> ({c.credit.source})</figcaption>
                  : <figcaption>True class <b>{className(c.truth)}</b></figcaption>)}
              </figure>
              <div className={`pl-layer${bars ? " on" : ""}`}>
                {c && <SoftmaxBars ranked={ranked(pos.stage === "model" ? c.raw : c.tempered)}
                  state={kept ? bucketState(c.outcome) as StateId : c.outcome === "reaim" && pos.stage === "visor" ? "cannot_assess" : "unrated"}
                  answer={kept ? c.set : []} threshold={kept ? cut : null} />}
              </div>
              <div className={`pl-layer${c && pos.stage === "ood" ? " on" : ""}`}>{c && <PLines c={c} alpha={PIPE.served.ood} />}</div>
              {detail && c && (
                <div className="pl-detail" role="dialog" aria-label="Stage detail">
                  <button type="button" className="pl-close" aria-label="Close" onClick={() => setDetail(null)}>×</button>
                  {detail === "knn" && <KnnHist knn={PIPE.knn} at={c.knn_dist} alpha={PIPE.served.ood} />}
                  {detail === "aon" && <AonScatter aon={PIPE.aon} at={PIPE.aon.crops[pos.crop]} alpha={PIPE.served.ood} />}
                  {detail === "set" && <ClassCoverage cov={cell.class_coverage} target={1 - PIPE.served.cp} />}
                </div>
              )}
            </div>
          </div>

          <ul className={`pl-buckets${done ? " is-done" : ""}`} aria-label="The visor's buckets">
            {BUCKETS.filter(([o]) => o !== "unclear").map(([o, st, label]) => {
              const i = PIPE.crops.findIndex((x) => x.outcome === o);
              return (
                <li key={o} className={`pl-bucket tone-${st}`}>
                  <span className="pl-bucket-head"><StateGlyph state={st} size={15} /> {label}</span>
                  <span className="pl-slot" ref={(el) => { slots.current[o] = el; }}>
                    {i >= 0 && <img src={PIPE.crops[i].image} alt="" className={landed(i) ? "in" : undefined} />}
                  </span>
                  <span className="pl-share"><span className="fig">{pct(share[o], 0)}</span> of test crops</span>
                  {i >= 0 && PIPE.crops[i].credit && <a className="pl-credit" href={PIPE.crops[i].credit!.url} target="_blank" rel="noreferrer">Glove photo: {PIPE.crops[i].credit!.author}, {PIPE.crops[i].credit!.license}</a>}
                </li>
              );
            })}
          </ul>
          <div className="pl-thumb" ref={thumb} aria-hidden="true">{c && <img src={c.image} alt="" />}</div>
        </div>
      </div>
    </div>
  );
}

/** What the current stage does, in one heading and one or two sentences, with B's own numbers. */
function Note({ pos, c, rep, cut }: { pos: ReturnType<typeof where>; c: PipeCrop | null; rep: Report; cut: number }) {
  const a = rep.arch_a, s = a.by_alpha[a.alpha.toFixed(2)], held = a.gate_tests?.sets.find((x) => x.set === "off-target held out");
  if (pos.crop < 0) return <><strong>Five crops, five answers</strong><p>Scroll to send them through. The model, a check, a temperature and a conformal set decide what the visor shows.</p></>;
  if (!c) return <><strong>One crop in every bucket</strong><p>Each of the five took its own path. Of the {int(PIPE.test_half.n)} test crops, {pct(s.cp_coverage)} get an answer that holds their true class, and {pct(s.human_attention, 0)} go to a person.</p></>;
  const n = pos.crop + 1, top = ranked(c.raw)[0], stopped = c.p_ood <= PIPE.served.ood;
  const words = c.set.filter((x) => x !== "no-defect").map((x) => className(x).toLowerCase());
  switch (pos.stage) {
    case "crop": return isOff(c)
      ? <><strong>Crop {n} of 5: {ABOUT[c.outcome]}</strong><p>A photo from the team's non-aircraft set, cut to the same 224 px square. The model was never trained on anything like it, but it still gives every class a probability.</p></>
      : <><strong>Crop {n} of 5: {ABOUT[c.outcome]}</strong><p>A 224 px square from the test photos. Its true class is {className(c.truth).toLowerCase()}.</p></>;
    case "model": return <><strong>The model's first guess: {className(top.cls)} {pct(top.p, 0)}</strong><p>Eight rotated and flipped views, a ResNet-18, five MLP heads and a vote of the 5 nearest training crops give a probability for each of the 7 classes. On the test crops: macro-F1 <span className="fig">{rep.test.macro_f1.toFixed(3)}</span>.</p></>;
    case "ood": return <><strong>{stopped ? "Stopped: this looks unfamiliar" : "Passed: this looks like aircraft skin"}</strong><p>Two conformal p-values: how close the crop is to its 5 nearest training crops, and how much it looks like aircraft skin rather than people, tools or scenes. Twice the smaller must be above {PIPE.served.ood}. It stops {pct(s.actions["re-aim"])} of normal test crops{held ? <> and {pct(held.stopped, 0)} of non-aircraft photos it never saw</> : null}.</p></>;
    case "temp": return <><strong>Softened percentages</strong><p>Every probability is softened with one temperature, fitted with equal total weight per validation photo family. For this model it changes little (calibration error <span className="fig">{a.ece.raw.toFixed(3)}</span> → <span className="fig">{a.ece.tempered.toFixed(3)}</span>); stated percentages still run a little high. The ranking never changes, and the guarantee comes from the next step, not the percentage.</p></>;
    case "set": return <><strong>{c.set.length === 1 ? `One class passes: ${className(c.set[0])}` : `${c.set.length} classes pass the cut-off`}</strong><p>Conformal prediction keeps every class at {pct(cut, 1)} or more. The cut-off is set on {a.cal.families} calibration photo groups so the answer holds the true class {pct(1 - a.alpha, 0)} of the time; on the test crops it does {pct(s.cp_coverage)}, with {PIPE.grid[cellKey(PIPE.served.ood, PIPE.served.cp)].set_size.toFixed(2)} classes on average.</p></>;
    default: return <>
      <strong className={`pl-verdict tone-${bucketState(c.outcome)}`}><StateGlyph state={bucketState(c.outcome)} size={20} /> {c.outcome === "show" ? `${className(c.set[0])} ${pct(c.tempered[PIPE.classes.indexOf(c.set[0])], 0)}`
        : c.outcome === "ambiguous" ? `Ambiguous: ${words.join(" or ")}` : c.outcome === "possible" ? `Possible ${words.join(" or ")}: check`
        : c.outcome === "nothing" ? "Nothing shown" : "Can't assess: re-aim"}</strong>
      <p>{c.outcome === "show" ? "One class is left, so the visor names it." : c.outcome === "ambiguous" ? "Two defect classes are left, so a person decides."
        : c.outcome === "possible" ? "A defect and no defect are both left, so the visor asks a person to check."
        : c.outcome === "nothing" ? "Only no defect is left, so the visor stays blank." : isOff(c) ? "The check stopped it before any class was shown, so the model's guess never reaches the visor." : "The check stopped it before any class was shown."} {pct(c.outcome === "show" ? s.actions["show class"] : c.outcome === "ambiguous" ? s.actions.ambiguous : c.outcome === "possible" ? s.actions["possible defect"] ?? 0 : c.outcome === "nothing" ? s.actions["nothing shown"] ?? 0 : s.actions["re-aim"], 0)} of test crops end here.</p>
    </>;
  }
}

/** The crop's two p-values and their Bonferroni union, on a log scale, against where each one stops a crop. */
function PLines({ c, alpha }: { c: PipeCrop; alpha: number }) {
  const rows: [string, number, number][] = [["Near its 5 training neighbours", c.p_knn, alpha / 2], ["Looks like aircraft skin", c.p_aon, alpha / 2],
    ["Combined: twice the smaller", c.p_ood, alpha]];
  return (
    <div className="pl-plines">
      {rows.map(([label, p, a], i) => (
        <div key={label} className={`pl-pline${i === 2 ? " is-total" : ""}${p <= a ? " is-stop" : ""}`}>
          <span>{label}</span>
          <span className="pl-ptrack" style={{ "--cut": px(a), "--at": px(p) } as React.CSSProperties}>
            <i className="pl-pzone" /><i className="pl-pcut" /><i className="pl-pdot" />
          </span>
          <span className="fig">p {fmtP(p)}</span>
        </div>
      ))}
      <p className="pl-pkey"><i className="pl-pzone-key" /> Stopped left of the line (p ≤ {alpha / 2} for one part, {alpha} combined) · log scale, 0.005 to 1</p>
    </div>
  );
}
