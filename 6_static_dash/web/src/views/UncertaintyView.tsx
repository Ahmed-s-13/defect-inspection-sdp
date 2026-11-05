import type React from "react";
import { motion } from "motion/react";
import { useEffect, useState, type ReactNode } from "react";
import { getReport, type Bin, type Report } from "../api/client";
import { AlphaLab } from "../components/AlphaLab";
import { AnimatedNumber } from "../components/AnimatedNumber";
import { AonExplorer, type AonData } from "../components/charts/AonExplorer";
import { PIPE_AT, PipelineScene } from "../components/PipelineScene";
import { RuleDraw } from "../components/RuleDraw";
import u from "../data/uncertainty.json";
import cpComparison from "../data/cp_comparison.json";
import oodComparison from "../data/ood_comparison.json";
import experiments from "../data/uncertainty_experiments.json";
import falseRejection from "../data/false_rejection.json";
import { className, int, pct } from "../lib/format";
import { useReveal } from "../lib/useReveal";
import { usePresent } from "../state/present";
import { Confidence, Refusals } from "./ModelView";
import "../styles/model.css";
import "../styles/uncertainty.css";

/** scripts/uncertainty_data.py: coverage by class with an example each way, and the aircraft-or-not chart. */
interface ClassRow { cls: string; n: number; coverage: number; coverage_passed: number; in_set_right: number; alone_right: number; alone_n: number; groups: number }
const U = u as unknown as {
  overall: { n: number; groups: number; coverage: number; coverage_passed: number; confidence_passed: number; single: number; set_size: number; single_right: number; multi_right: number };
  by_class: ClassRow[]; aon: AonData & { n_pos: number; n_neg: number }; gates: { set: string; label: string; stopped: number }[];
  by_set_size: { size: number; n: number; correct: number; coverage: number | null }[];
};
const ease = [0.22, 1, 0.36, 1] as const;
const iv = (i: number) => ({ "--i": i } as React.CSSProperties);
/** The experiment reports store rates as text with their ranges ("1.4% (0.0–3.7%)"); the charts use the mean. */
const lead = (s: string) => parseFloat(s) / 100;

/** Static dash's Uncertainty page: the same story as the dashboard's, told shorter. The results come first;
 *  the experiments behind each design choice are one chart each instead of a table and its notes. */
export function UncertaintyView() {
  const [rep, setRep] = useState<Report | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { getReport().then(setRep).catch((e: Error) => setErr(e.message)); }, []);
  const present = usePresent();
  if (err) return <div className="model unc"><section className="model-empty"><h2 className="view-title">Uncertainty</h2><p>{err}</p></section></div>;
  if (!rep) return <div className="model unc"><section className="model-empty"><h2 className="view-title">Uncertainty</h2><p className="muted">Loading the report…</p></section></div>;

  const sections: [string, ReactNode][] = [
    ["Intro", <Intro rep={rep} />], ["Coverage", <Coverage />], ["Percentages", <Percentages rep={rep} />],
    ["Confidence", <Confidence rep={rep} />], ["AoN", <AonSection rep={rep} />], ["Refusals", <Refusals rep={rep} />],
    ["Choices", <Choices />], ["Pipeline", null], ["Alphas", <AlphaLab rep={rep} />],
  ];
  const sizes = sections.map(([name]) => (name === "Pipeline" ? PIPE_AT.length : 1));
  const firsts = sizes.map((_, i) => sizes.slice(0, i).reduce((a, b) => a + b, 0));
  const dimAt = (step: number) => present.on && present.step !== step;
  return (
    <div className="model unc">
      {sections.map(([name, node], i) => (name === "Pipeline"
        ? <section key={name} className="model-section"><PipelineScene rep={rep} first={firsts[i]} dimAt={dimAt} /></section>
        : <motion.section key={name} data-step={firsts[i]} className="model-section" animate={{ opacity: dimAt(firsts[i]) ? 0.25 : 1 }} transition={{ duration: 0.3 }}>{node}</motion.section>))}
      <footer className="provenance"><span>Main results: {int(U.overall.n)} held-out test crops ({U.overall.groups} photo groups) that calibrated nothing. The design-choice charts come from separate calibration-stage experiments and OOD evaluations. A prototype decision aid, not an approved inspection tool.</span></footer>
    </div>
  );
}

function Intro({ rep }: { rep: Report }) {
  const rv = useReveal<HTMLDivElement>(0.9);
  const s = rep.arch_a.by_alpha[rep.arch_a.alpha.toFixed(2)];
  return (
    <div className="unc-intro" ref={rv.ref} data-shown={rv.shown}>
      <h1 className="view-title rv" style={iv(0)}>Uncertainty</h1>
      <p className="view-sub rv" style={iv(1)}>How the helmet decides what to show, what to hedge, and what to hand to a person. The target: the answer holds the true class 90% of the time (α = 0.10).</p>
      <p className="unc-lead rv" style={iv(2)}><span className="fig unc-big"><AnimatedNumber value={rv.shown ? U.overall.coverage * 100 : 0} format={(v) => `${v.toFixed(1)}%`} /></span>
        of {int(U.overall.n)} held-out test crops got an answer that holds the true class. Showing only the top class, {pct(1 - s.accuracy)} would be wrong with no warning; with these answers it is {pct(s.single_class_wrong)}, and {pct(s.human_attention, 0)} go to a person.</p>
      <dl className="unc-says unc-outcomes rv" style={iv(3)}>
        <div><dt>Get a single-class answer</dt><dd><strong className="unc-right">{pct(U.overall.single, 1)}</strong><small>of crops that pass the unfamiliar-input check</small></dd></div>
        <div><dt>Single-class answers are right</dt><dd><strong className="unc-right">{pct(U.overall.single_right, 1)}</strong><small>of those answers</small></dd></div>
        <div><dt>Answers of 2+ classes hold the true one</dt><dd><strong className="unc-right">{pct(U.overall.multi_right, 1)}</strong><small>of those answers</small></dd></div>
      </dl>
      <SetSizeStrip />
    </div>
  );
}

/** How big the answers are: one bar per set size, as a share of the crops that passed the gate. */
function SetSizeStrip() {
  const rv = useReveal<HTMLDivElement>(0.85);
  const rows = U.by_set_size.filter((r) => r.n > 0);
  const total = rows.reduce((a, r) => a + r.n, 0), max = Math.max(...rows.map((r) => r.n));
  return (
    <div className="unc-sizes rv" style={iv(4)} ref={rv.ref} data-shown={rv.shown}>
      <p className="unc-bars-head">How big the answers are, on the {int(total)} test crops that passed the unfamiliar-input check</p>
      <ol className="unc-size-cols" style={{ "--n": rows.length } as React.CSSProperties}>
        {rows.map((r, i) => (
          <li key={r.size}>
            <span className="fig unc-size-share">{pct(r.n / total, 0)}</span>
            <span className="unc-size-bar"><motion.i initial={false} animate={{ scaleY: rv.shown ? r.n / max : 0 }} transition={{ duration: rv.t(0.7), delay: rv.t(0.06) * i, ease }} /></span>
            <span className="fig unc-size-n">{r.size}</span>
            <small>{r.coverage === null ? "—" : `${pct(r.coverage, 0)} right`}</small>
          </li>
        ))}
      </ol>
      <p className="unc-bars-key">Classes in the answer, with how often an answer of that size held the true class</p>
    </div>
  );
}

function Coverage() {
  const rv = useReveal<HTMLDivElement>(0.8);
  const rows = [...U.by_class].sort((a, b) => a.coverage - b.coverage);
  const lo = rows[0], hi = rows[rows.length - 1];
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <div className="unc-cov">
        <div>
          <h2 className="panel-title rv" style={iv(0)}>How often the answer includes the right class</h2>
          <p className="panel-sub rv" style={iv(1)}>About 9 in 10 overall ({pct(U.overall.coverage)}), but not equally for every class: from {pct(lo.coverage, 0)} of real {className(lo.cls).toLowerCase()} crops to {pct(hi.coverage, 0)} of real {className(hi.cls).toLowerCase()} crops. The 90% target is for all crops together, not for each class.</p>
        </div>
        <div className="rv" style={iv(2)}>
          <p className="unc-bars-head">Real crops of this class whose answer included it</p>
          <ol className="unc-bars" aria-label="Real crops of each class whose answer included it">
            {rows.map((r, i) => (
              <li key={r.cls} className="unc-bar">
                <span>{className(r.cls)}</span>
                <span className="unc-track"><motion.i initial={false} animate={{ scaleX: rv.shown ? r.coverage : 0 }} transition={{ duration: rv.t(0.8), delay: rv.t(0.06) * i, ease }} className={r.coverage < 0.9 ? "is-low" : undefined} /><b /></span>
                <span className="fig">{pct(r.coverage, 0)}</span>
              </li>
            ))}
          </ol>
          <p className="unc-bars-key"><i aria-hidden="true" /> 90% line: the overall target</p>
        </div>
      </div>
    </div>
  );
}

const agg = (bins: Bin[], lo: number, hi: number) => {
  const b = bins.filter((x) => x.lo >= lo - 1e-9 && x.hi <= hi + 1e-9 && x.n > 0);
  const n = b.reduce((a, x) => a + x.n, 0);
  return { n, said: b.reduce((a, x) => a + x.confidence * x.n, 0) / n, right: b.reduce((a, x) => a + x.accuracy * x.n, 0) / n };
};

function Percentages({ rep }: { rep: Report }) {
  const rv = useReveal<HTMLDivElement>(0.8);
  const t = rep.arch_a.reliability.tempered;
  const rows: [string, ReturnType<typeof agg>][] = [["Says 90% or more", agg(t, 0.9, 1)], ["Says 80–90%", agg(t, 0.8, 0.9)], ["Says 50–70%", agg(t, 0.5, 0.7)]];
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>What the percentages mean</h2>
      <p className="panel-sub rv" style={iv(1)}>The % is the model's confidence, softened by one temperature. Over many crops it nearly matches how often the model is right (an average gap of {(rep.arch_a.ece.tempered * 100).toFixed(1)} points), but runs a little high. It is a belief, not a promise about one crop.</p>
      <dl className="unc-says">
        {rows.map(([label, r], i) => (
          <div key={label} className="rv" style={iv(2 + i)}>
            <dt>{label}</dt>
            <dd><span className="fig unc-said">{pct(r.said, 0)}</span><span className="unc-arrow" aria-hidden="true">→</span><span className="fig unc-right"><AnimatedNumber value={rv.shown ? r.right * 100 : 0} format={(v) => `${v.toFixed(0)}%`} /></span>
              <small>stated → right, on {int(r.n)} test crops · {r.right < r.said - 0.02 ? "over-confident" : r.right > r.said + 0.02 ? "under-confident" : "about right"}</small></dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function AonSection({ rep }: { rep: Report }) {
  const rv = useReveal<HTMLDivElement>(0.75);
  const s = rep.arch_a.by_alpha[rep.arch_a.alpha.toFixed(2)];
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>How the aircraft-or-not check works</h2>
      <p className="panel-sub rv" style={iv(1)}>It catches what the model was never meant to see: people, tools, sky. Together with a distance check (kNN), it wrongly stops {pct(s.actions["re-aim"])} of normal test crops.</p>
      <ol className="unc-steps rv" style={iv(2)}>
        <li><b>Features.</b> The model turns every crop into 768 numbers.</li>
        <li><b>A learned line.</b> A logistic regression separates {int(U.aon.n_pos)} aircraft-skin crops from {int(U.aon.n_neg)} crops of people, tools, vehicles and scenes. Past the line it leans "not aircraft".</li>
        <li><b>A p-value.</b> The score is compared with 88 calibration photo groups. If this crop is too unusual for aircraft skin (each check's p-value at or below {rep.arch_a.alpha_ood / 2}), the visor says re-aim.</li>
      </ol>
      <div className="rv" style={iv(3)}><AonExplorer d={U.aon} shown={rv.shown} /></div>
      <figure className="unc-false-reject rv" style={iv(4)}>
        <img src={falseRejection.image} width={160} height={160} loading="lazy" alt="D1 corrosion crop from family 533, incorrectly rejected by the aircraft-or-not check" />
        <figcaption><strong>When it gets it wrong.</strong> This test crop is real corrosion, but the check found it too unusual (p = {falseRejection.p_aon.toFixed(4)}) and the visor said re-aim. A re-aim does not mean there is no defect.</figcaption>
      </figure>
    </div>
  );
}

/* ---------- the design choices, one chart each ---------- */

interface Bar { label: string; v: number; ours?: boolean; text?: string }
/** Horizontal bars for one measure; ours is drawn in ink, the alternatives in grey. */
function Bars({ title, rows, shown, max, digits = 1 }: { title: string; rows: Bar[]; shown: boolean; max?: number; digits?: number }) {
  const top = max ?? Math.max(...rows.map((r) => r.v));
  return (
    <div className="ch-bars">
      <p className="ch-measure">{title}</p>
      <ol>
        {rows.map((r, i) => (
          <li key={r.label} className={r.ours ? "is-ours" : undefined}>
            <span className="ch-label">{r.label}</span>
            <span className="ch-track"><motion.i initial={false} animate={{ scaleX: shown ? Math.max(r.v / top, 0.004) : 0 }} transition={{ duration: 0.7, delay: 0.05 * i, ease }} /></span>
            <span className="fig ch-val">{r.text ?? pct(r.v, digits)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Choice({ kept, q, children, why, note, i }: { kept: string; q: string; children: ReactNode; why: string; note?: string; i: number }) {
  return (
    <article className="ch-card rv" style={iv(2 + i)}>
      <p className="ch-kept">{kept}</p>
      <h3>{q}</h3>
      {children}
      <p className="ch-why">{why}</p>
      {note && <p className="ch-note">{note}</p>}
    </article>
  );
}

const HIST = "Calibration-stage experiment at the earlier OOD α of 0.08.";

function Choices() {
  const rv = useReveal<HTMLDivElement>(0.6);
  const sh = rv.shown;
  const ood = [...oodComparison.variants].filter((v) => ["ResNet + AoN + kNN", "ResNet + AoN", "DINOv2 + AoN", "DINOv2 + kNN"].includes(v.name))
    .sort((a, b) => a.d1_frr - b.d1_frr);
  const gates = experiments.gate_combinations.map((g, k) => ({ ...g, label: k === 0 ? "Bonferroni" : g.name.replace("kNN + AoN, ", ""), ours: k === 0 }));
  const alphas = experiments.ood_alpha.slice(0, 4).map((a) => ({ ...a, ours: Number(a.alpha) === oodComparison.alpha_ood }));
  const cals = experiments.calibrators.map((c) => ({ label: c.name === "none" ? "No softening" : c.name.replace(" (B)", ""), v: parseFloat(c.ece), ours: c.name.startsWith("temperature") }));
  const fw = cpComparison.variants;
  const guard = [experiments.guard[0], experiments.guard[1]];
  const split = experiments.calibration_split.slice(0, 2);
  return (
    <div className="panel" ref={rv.ref} data-shown={sh}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>The choices behind it, and what we tested</h2>
      <p className="panel-sub rv" style={iv(1)}>Each setting was compared against alternatives before we fixed it. We put a low false-rejection rate first: usable aircraft crops should reach the model. Ink is what we kept.</p>
      <div className="ch-grid">
        <Choice i={0} kept="ResNet + AoN + kNN" q="Which unfamiliar-input check?" why="Ours stops the fewest valid D1 crops. DINOv2 + AoN lets fewer non-aircraft crops through, but stops almost twice as many good ones, so no option wins on every measure."
          note={`OOD α ${oodComparison.alpha_ood} · ${int(oodComparison.test_crops)} D1 and ${int(oodComparison.offtarget_crops)} non-aircraft held-out crops.`}>
          <Bars shown={sh} digits={1} title="Valid D1 crops wrongly stopped (lower is better)" rows={ood.map((v) => ({ label: v.name, v: v.d1_frr, ours: v.selected }))} />
          <Bars shown={sh} digits={1} title="Non-aircraft crops let through (lower is better)" rows={ood.map((v) => ({ label: v.name, v: v.far, ours: v.selected }))} />
        </Choice>

        <Choice i={1} kept={`α = ${oodComparison.alpha_ood}`} q="How strict should the check be?" why="A stricter cutoff stops more non-aircraft inputs, but also more real defects. We moved from 0.08 to 0.05 to stop fewer real defects, accepting that more odd inputs pass."
          note="Calibration-stage sweep over 200 family halvings. Wallpaper is a stress test of inputs the check never saw.">
          <Bars shown={sh} title="Real defects wrongly stopped" rows={alphas.map((a) => ({ label: `α ${a.alpha}`, v: lead(a.defect_frr), ours: a.ours }))} />
          <Bars shown={sh} title="Wallpaper photos let through" rows={alphas.map((a) => ({ label: `α ${a.alpha}`, v: lead(a.wallpaper_far), ours: a.ours }))} />
        </Choice>

        <Choice i={2} kept="Bonferroni" q="How to combine the two checks?" why="Bonferroni doubles the smaller p-value and is valid even though both checks share ResNet features. Simes shows slightly fewer non-aircraft photos but stops more real defects." note={HIST}>
          <Bars shown={sh} title="Real defects wrongly stopped" rows={gates.map((g) => ({ label: g.label, v: lead(g.defect_frr), ours: g.ours }))} />
          <Bars shown={sh} title="Non-aircraft photos shown as a defect" rows={gates.map((g) => ({ label: g.label, v: lead(g.offtarget_shown), ours: g.ours }))} />
        </Choice>

        <Choice i={3} kept="One temperature" q="How to soften the percentages?" why="Isotonic measured a little lower, but its improvement was within the noise (the interval includes zero). One temperature is simpler and never changes which class wins."
          note="Calibration error, fitted on validation crops and measured on the calibration half.">
          <Bars shown={sh} title="Calibration error (lower is better)" rows={cals.map((c) => ({ ...c, text: c.v.toFixed(3) }))} />
        </Choice>

        <Choice i={4} kept="Family weighting" q="Should every crop count the same?" why="Hundreds of crops can come from one photo. Weighting each photo family equally keeps one busy photo from setting the cutoff, and holds coverage near the 90% target."
          note={`${int(cpComparison.test_crops)} held-out test crops, both cutoffs fitted on the same ${int(cpComparison.calibration_crops)} calibration crops.`}>
          <Bars shown={sh} max={1} title="Answers holding the true class (target 90%)" rows={[
            { label: "Family-weighted", v: fw.family_weighted.coverage, ours: true }, { label: "Every crop equal", v: fw.equal_crop.coverage }]} />
          <Bars shown={sh} max={1} title="Same, averaged so each class counts equally" rows={[
            { label: "Family-weighted", v: fw.family_weighted.macro_coverage, ours: true }, { label: "Every crop equal", v: fw.equal_crop.macro_coverage }]} />
        </Choice>

        <Choice i={5} kept="Guard off" q="Should a lone “no defect” be double-checked?" why="The strictest guard did reach zero silent misses, but only by never letting the visor stay silent, so every clean panel raised a message. Looser settings changed nothing." note={HIST}>
          <Bars shown={sh} max={0.02} title="Real defects the visor stayed silent on" rows={guard.map((g, k) => ({ label: k ? "Guard at 0.02" : "No guard", v: lead(g.silent_miss), ours: !k }))} />
          <Bars shown={sh} max={1} title="Clean crops the visor stayed silent on" rows={guard.map((g, k) => ({ label: k ? "Guard at 0.02" : "No guard", v: lead(g.clean_silence), ours: !k }))} />
        </Choice>

        <Choice i={6} kept="All families for training" q="Would a bigger calibration set help?" why="Setting aside 20% of training families for calibration cut false alarms on clean crops, but silent misses rose and the model had to change too, so it is not a clean win." note={HIST}>
          <Bars shown={sh} title="Real defects the visor stayed silent on" rows={split.map((r, k) => ({ label: k ? "166 cal families" : "44 cal families", v: lead(r.silent_miss), ours: !k }))} />
          <Bars shown={sh} title="Clean crops given a message" rows={split.map((r, k) => ({ label: k ? "166 cal families" : "44 cal families", v: lead(r.false_alarm), ours: !k }))} />
        </Choice>
      </div>
    </div>
  );
}
