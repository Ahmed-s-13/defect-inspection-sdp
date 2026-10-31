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
    ["Intro", <Intro rep={rep} />], ["Coverage", <Coverage />], ["Set sizes", <SetSizes />], ["Percentages", <Percentages rep={rep} />],
    ["Confidence", <Confidence rep={rep} />], ["AoN", <AonSection rep={rep} />], ["OOD comparison", <OodComparison />], ["Refusals", <Refusals rep={rep} />],
    ["Family weighting", <FamilyWeighting />], ["Pipeline", null], ["Alphas", <AlphaLab rep={rep} />],
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
  const confidence = agg(rep.arch_a.reliability.tempered, 0, 1).said;
  const macroCoverage = U.by_class.reduce((sum, row) => sum + row.coverage, 0) / U.by_class.length;
  return (
    <div className="unc-intro" ref={rv.ref} data-shown={rv.shown}>
      <h1 className="view-title rv" style={iv(0)}>Uncertainty</h1>
      <p className="view-sub rv" style={iv(1)}>Calibration and conformal prediction: how the helmet decides what to show, what to hedge, and what to hand to a person, with a 90% target coverage (nominal coverage level, α = 0.10).</p>
      <p className="unc-lead rv" style={iv(2)}><span className="fig unc-big"><AnimatedNumber value={rv.shown ? U.overall.coverage * 100 : 0} format={(v) => `${v.toFixed(1)}%`} /></span>
        observed overall coverage: the candidate set includes the true class on {pct(U.overall.coverage)} of {int(U.overall.n)} held-out test crops, including OOD-rejected crops. Showing only the top class, {pct(1 - s.accuracy)} would be wrong with no warning; with these answers it is {pct(s.single_class_wrong)}, and {pct(s.human_attention, 0)} go to a person.</p>
      <p className="unc-results-note rv" style={iv(2)}><strong className="fig">{pct(macroCoverage, 2)} macro coverage</strong> — the average of the seven class coverages, giving each class equal weight. Overall coverage gives each crop equal weight. These are observed results, not a guarantee for every class or image.</p>
      <div className="rv" style={iv(3)}>
        <h2 className="panel-title">After the unfamiliar-input check</h2>
        <p className="unc-results-note">Measured only on test crops accepted by the OOD gate; single-class sets include no-defect.</p>
        <dl className="unc-says unc-outcomes">
          <div><dt>Receive a single-class set</dt><dd><strong className="unc-right">{pct(U.overall.single, 2)}</strong><small>of accepted crops</small></dd></div>
          <div><dt>Single-class sets are correct</dt><dd><strong className="unc-right">{pct(U.overall.single_right, 2)}</strong><small>of single-class sets</small></dd></div>
          <div><dt>Multiple-class sets hold the true class</dt><dd><strong className="unc-right">{pct(U.overall.multi_right, 2)}</strong><small>of multiple-class sets</small></dd></div>
        </dl>
      </div>
      <dl className="unc-words">
        <div className="rv" style={iv(4)}><dt>Confidence</dt><dd>The model's probability for the class it shows, the % on the visor. A belief, not a promise.</dd><dd className="unc-average"><strong className="fig">{pct(U.overall.confidence_passed, 2)}</strong> mean calibrated top-class probability after OOD rejection, including no-defect predictions; {pct(confidence, 2)} across all test crops.</dd><dd>Accepted crops proceed to temperature scaling and then the conformal set. The OOD gate uses features, not these calibrated probabilities.</dd></div>
        <div className="rv" style={iv(5)}><dt>Coverage</dt><dd>How often the answer, one class or several, holds the true class. Conformal prediction aims it at 90% on average.</dd><dd className="unc-average"><strong className="fig">{pct(U.overall.coverage_passed, 2)}</strong> of accepted crops; {pct(U.overall.coverage, 2)} across all test crops.</dd></div>
        <div className="rv" style={iv(6)}><dt>Referral</dt><dd>How often the visor hands the crop to a person: ambiguous, possible, or re-aim.</dd><dd className="unc-average"><strong className="fig">{pct(s.human_attention, 2)}</strong> of all test crops go to a person.</dd></div>
      </dl>
    </div>
  );
}

function FamilyWeighting() {
  const rv = useReveal<HTMLDivElement>(0.8);
  const { family_weighted: weighted, equal_crop: equal } = cpComparison.variants;
  const rows: [string, number, number, boolean?][] = [
    ["Probability cutoff", weighted.probability_cutoff, equal.probability_cutoff],
    ["Overall coverage", weighted.coverage, equal.coverage],
    ["Macro coverage (each class equal)", weighted.macro_coverage, equal.macro_coverage],
    ["Family-average coverage (each family equal)", weighted.family_coverage, equal.family_coverage],
    ["Average set size", weighted.mean_set_size, equal.mean_set_size, true],
    ["Single-class sets", weighted.singleton_rate, equal.singleton_rate],
  ];
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>Why we use family weighting</h2>
      <p className="panel-sub rv" style={iv(1)}>We use family weighting so that hundreds of copies of one photo do not influence the calibration cutoff more than a different photo with only a few crops.</p>
      <div className="unc-table-wrap rv" style={iv(2)}>
        <table className="unc-size-table">
          <caption>After OOD rejection · α = {cpComparison.alpha.toFixed(2)} · {int(cpComparison.test_crops)} held-out test crops from {cpComparison.test_families} families</caption>
          <thead><tr><th scope="col">Metric</th><th scope="col">Family-weighted CP (current)</th><th scope="col">Default CP (equal crop weights)</th></tr></thead>
          <tbody>{rows.map(([label, a, b, isSize]) => <tr key={label}><th scope="row">{label}</th><td>{isSize ? a.toFixed(2) : pct(a, 2)}</td><td>{isSize ? b.toFixed(2) : pct(b, 2)}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="unc-results-note rv" style={iv(3)}>Same model, temperature scaling and OOD gate. Each cutoff was fitted separately on the same {int(cpComparison.calibration_crops)} accepted calibration crops. Equal crop weights produce smaller sets but lower coverage on this test split; these are observed results, not a guarantee for future inputs.</p>
    </div>
  );
}

function SetSizes() {
  const rv = useReveal<HTMLDivElement>(0.8);
  const total = U.by_set_size.reduce((sum, row) => ({ n: sum.n + row.n, correct: sum.correct + row.correct }), { n: 0, correct: 0 });
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>Coverage by prediction-set size</h2>
      <p className="panel-sub rv" style={iv(1)}>At α = 0.10, after OOD rejection. A set is correct when it contains the true label; no-defect counts as one class.</p>
      <div className="unc-table-wrap rv" style={iv(2)}>
        <table className="unc-size-table">
          <caption>Held-out crops accepted by the unfamiliar-input gate</caption>
          <thead><tr><th scope="col">Set size</th><th scope="col">Sets</th><th scope="col">Contain true label</th><th scope="col">Coverage</th></tr></thead>
          <tbody>{U.by_set_size.map((row) => <tr key={row.size}><th scope="row">{row.size}</th><td>{int(row.n)}</td><td>{row.n ? int(row.correct) : "—"}</td><td>{row.coverage === null ? "—" : pct(row.coverage, 2)}</td></tr>)}</tbody>
          <tfoot><tr><th scope="row">Total</th><td>{int(total.n)}</td><td>{int(total.correct)}</td><td>{pct(total.correct / total.n, 2)}</td></tr></tfoot>
        </table>
      </div>
      <p className="unc-results-note rv" style={iv(3)}>Observed coverage, not a guarantee for each size. The size-6 result comes from only one crop. Empty sets retain the top class, so size 0 does not occur; rejected inputs are excluded, not counted as empty sets.</p>
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
          <p className="panel-sub rv" style={iv(1)}>About 9 in 10 candidate sets include the right class ({pct(U.overall.coverage)}), but not equally for every class. Each bar starts from all test crops that really are that class, including OOD-rejected crops, and counts how often their candidate set included it: from {pct(lo.coverage, 0)} of real {className(lo.cls).toLowerCase()} crops to {pct(hi.coverage, 0)} of real {className(hi.cls).toLowerCase()} crops. The 90% target is overall, not per class or image; our family-weighted, OOD-filtered implementation is evaluated empirically here. Among OOD-accepted crops, single-class sets were correct {pct(U.overall.single_right)} of the time; sets with two or more classes included the right one {pct(U.overall.multi_right)}.</p>
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

function OodComparison() {
  const rv = useReveal<HTMLDivElement>(0.8);
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>Why we keep ResNet + AoN + kNN</h2>
      <p className="panel-sub rv" style={iv(1)}>We prioritize a low false rejection rate (FRR) above reducing false acceptance further: usable aircraft crops should reach the inspection model. ResNet + AoN + kNN had the lowest D1 and D5 FRR among these tested configurations, so our deployed OOD gate stays unchanged. We accept the higher held-out false acceptance rate (FAR) shown in this comparison.</p>
      <div className="unc-table-wrap rv" style={iv(2)}>
        <table className="unc-size-table">
          <caption>Architecture B · quality gate off · OOD α = {oodComparison.alpha_ood} · sorted by D1 FRR, lowest first · green marks our chosen configuration</caption>
          <thead><tr><th scope="col">OOD configuration</th><th scope="col">FAR</th><th scope="col">D1 FRR</th><th scope="col">D5 FRR</th></tr></thead>
          <tbody>{[...oodComparison.variants].sort((a, b) => a.d1_frr - b.d1_frr).map((row) => <tr key={row.name} className={row.selected ? "unc-ood-selected" : undefined}>
            <th scope="row">{row.name}</th>
            <td>{pct(row.far, 2)}</td><td>{pct(row.d1_frr, 2)}</td><td>{pct(row.d5_frr, 2)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <p className="unc-results-note rv" style={iv(3)}>FRR: valid aircraft crops stopped by the gate. FAR: off-target crops that pass it, regardless of what the visor later displays. Evaluated on {int(oodComparison.test_crops)} D1 crops, {int(oodComparison.d5_crops)} D5 crops and {int(oodComparison.offtarget_crops)} held-out off-target crops. Rates give each crop equal weight; no thresholds were tuned on these test results.</p>
      <p className="panel-sub rv" style={iv(4)}>AoN and DINOv2 have different jobs. AoN learns to distinguish aircraft crops from off-target examples; DINOv2 supplies image features and still needs a detector such as AoN or kNN. Our AoN uses features already produced by ResNet, avoiding a second image encoder. Combining it with kNN adds a learned off-target check to the distance check: ResNet kNN alone accepted {pct(oodComparison.variants[1].far, 2)} of off-target crops, compared with {pct(oodComparison.variants[2].far, 2)} for the combined gate.</p>
      <p className="unc-results-note rv" style={iv(5)}>For a fair comparison, every AoN above was trained on the same off-target categories, excluding the evaluation categories. The retained row tests our chosen configuration with that restricted training; the deployed AoN saw more categories, so these are comparison results rather than its shipped FAR. DINOv2 + AoN achieved the lowest FAR, but rejected more valid D1 and D5 crops.</p>
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
