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
import { className, int, pct } from "../lib/format";
import { useReveal } from "../lib/useReveal";
import { usePresent } from "../state/present";
import { Confidence, Refusals } from "./ModelView";
import "../styles/model.css";
import "../styles/uncertainty.css";

/** scripts/uncertainty_data.py: coverage by class with an example each way, and the aircraft-or-not chart. */
interface ClassRow { cls: string; n: number; coverage: number; coverage_passed: number; in_set_right: number; alone_right: number; alone_n: number; groups: number }
const U = u as unknown as {
  overall: { n: number; groups: number; coverage: number; coverage_passed: number; single: number; set_size: number; single_right: number; multi_right: number };
  by_class: ClassRow[]; aon: AonData & { n_pos: number; n_neg: number }; gates: { set: string; label: string; stopped: number }[];
};
const ease = [0.22, 1, 0.36, 1] as const;
const iv = (i: number) => ({ "--i": i } as React.CSSProperties);

/** Calibration + conformal prediction on one page: what the percentages mean, how often the answer holds the true
 *  class, how the helmet decides what to show, and what it refuses. Inferences first, then how. */
export function UncertaintyView() {
  const [rep, setRep] = useState<Report | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { getReport().then(setRep).catch((e: Error) => setErr(e.message)); }, []);
  const present = usePresent();
  if (err) return <div className="model unc"><section className="model-empty"><h2 className="view-title">Uncertainty</h2><p>{err}</p></section></div>;
  if (!rep) return <div className="model unc"><section className="model-empty"><h2 className="view-title">Uncertainty</h2><p className="muted">Computing the report…</p></section></div>;

  const sections: [string, ReactNode][] = [
    ["Intro", <Intro rep={rep} />], ["Coverage", <Coverage />], ["Percentages", <Percentages rep={rep} />],
    ["Confidence", <Confidence rep={rep} />], ["AoN", <AonSection rep={rep} />], ["Refusals", <Refusals rep={rep} />],
    ["Pipeline", null], ["Alphas", <AlphaLab rep={rep} />],
  ];
  const sizes = sections.map(([name]) => (name === "Pipeline" ? PIPE_AT.length : 1));
  const firsts = sizes.map((_, i) => sizes.slice(0, i).reduce((a, b) => a + b, 0));
  const total = sizes.reduce((a, b) => a + b, 0);
  const dimAt = (step: number) => present.on && present.step !== step;
  return (
    <div className="model unc">
      {sections.map(([name, node], i) => (name === "Pipeline"
        ? <section key={name} className="model-section"><PipelineScene rep={rep} first={firsts[i]} dimAt={dimAt} /></section>
        : <motion.section key={name} data-step={firsts[i]} className="model-section" animate={{ opacity: dimAt(firsts[i]) ? 0.25 : 1 }} transition={{ duration: 0.3 }}>{node}</motion.section>))}
      {present.on && <p className="step-hint">Section {present.step + 1} of {total} · ← → to move</p>}
      <footer className="provenance"><span>Measured on the {int(U.overall.n)} test crops ({U.overall.groups} photo groups) that calibrated nothing; data: scripts/uncertainty_data.py and pipeline_scene.py. A prototype decision aid, not an approved inspection tool.</span></footer>
    </div>
  );
}

function Intro({ rep }: { rep: Report }) {
  const rv = useReveal<HTMLDivElement>(0.9);
  const s = rep.arch_a.by_alpha[rep.arch_a.alpha.toFixed(2)];
  return (
    <div className="unc-intro" ref={rv.ref} data-shown={rv.shown}>
      <h1 className="view-title rv" style={iv(0)}>Uncertainty</h1>
      <p className="view-sub rv" style={iv(1)}>Calibration and conformal prediction: how the helmet decides what to show, what to hedge, and what to hand to a person.</p>
      <p className="unc-lead rv" style={iv(2)}><span className="fig unc-big"><AnimatedNumber value={rv.shown ? U.overall.coverage * 100 : 0} format={(v) => `${v.toFixed(1)}%`} /></span>
        of answers include the right class, on {int(U.overall.n)} test crops it never trained or calibrated on. Showing only the top class, {pct(1 - s.accuracy)} would be wrong with no warning; with these answers it is {pct(s.single_class_wrong)}, and {pct(s.human_attention, 0)} go to a person.</p>
      <dl className="unc-words">
        <div className="rv" style={iv(3)}><dt>Confidence</dt><dd>The model's probability for the class it shows, the % on the visor. A belief, not a promise.</dd></div>
        <div className="rv" style={iv(4)}><dt>Coverage</dt><dd>How often the answer, one class or several, holds the true class. Conformal prediction aims it at 90% on average.</dd></div>
        <div className="rv" style={iv(5)}><dt>Referral</dt><dd>How often the visor hands the crop to a person: ambiguous, possible, or re-aim.</dd></div>
      </dl>
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
          <p className="panel-sub rv" style={iv(1)}>About 9 in 10 answers include the right class ({pct(U.overall.coverage)}), but not equally for every class. Each bar starts from the crops that really are that class and counts how often their answer included it: from {pct(lo.coverage, 0)} of real {className(lo.cls).toLowerCase()} crops to {pct(hi.coverage, 0)} of real {className(hi.cls).toLowerCase()} crops. These are measured on the test crops, not promised: only the overall 90% is, and easy classes carry the hard ones. A single-class answer was right {pct(U.overall.single_right)} of the time; answers with two or more classes included the right one {pct(U.overall.multi_right)}.</p>
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
      <p className="panel-sub rv" style={iv(1)}>The % is the model's confidence, softened by one temperature. Over many crops it should match how often the model is right; it nearly does (calibration error {rep.arch_a.ece.tempered.toFixed(3)}, an average gap of {(rep.arch_a.ece.tempered * 100).toFixed(1)} points), but runs high at the top and in the middle. It describes the model's belief; it does not promise anything about a single crop.</p>
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
  const held = U.gates.find((g) => g.set === "off-target held out");
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>How the aircraft-or-not check works</h2>
      <p className="panel-sub rv" style={iv(1)}>It catches what the model was never meant to see. {held ? <>It stops {pct(held.stopped)} of off-target photos from categories it never trained on, </> : null}and wrongly stops {pct(s.actions["re-aim"])} of normal test crops.</p>
      <ol className="unc-steps rv" style={iv(2)}>
        <li><b>Features.</b> The model turns every crop into 768 numbers (its 8-view features).</li>
        <li><b>A learned line.</b> A logistic regression weighs those numbers: {int(U.aon.n_pos)} training crops of aircraft skin on one side, {int(U.aon.n_neg)} crops of people, tools, vehicles and scenes on the other (outlier exposure). Its score is the weighted sum plus a bias; past 0 it leans "not aircraft".</li>
        <li><b>A conformal p-value.</b> The score is compared with the scores of the 88 calibration photo groups: how unusual is this crop for aircraft skin? With the kNN distance's p-value, the smaller one doubled must stay above {rep.arch_a.alpha_ood}, or the visor says re-aim.</li>
      </ol>
      <div className="rv" style={iv(3)}><AonExplorer d={U.aon} shown={rv.shown} /></div>
    </div>
  );
}
