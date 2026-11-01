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
    ["Confidence", <Confidence rep={rep} />], ["AoN", <AonSection rep={rep} />], ["Gate combination", <GateCombination />], ["OOD threshold", <OodThreshold />], ["OOD comparison", <OodComparison />], ["Refusals", <Refusals rep={rep} />],
    ["Calibrator experiments", <CalibratorExperiments rep={rep} />], ["Miss guard", <MissGuardExperiment />], ["Calibration split", <CalibrationSplitExperiment />],
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
      <footer className="provenance"><span>Main held-out results: {int(U.overall.n)} test crops ({U.overall.groups} photo groups) that calibrated nothing. Calibration-stage experiments and external OOD evaluations use the separate samples described in their panels. Data: uncertainty_data.py, pipeline_scene.py, ood_comparison_data.py and uncertainty_experiments_data.py. A prototype decision aid, not an approved inspection tool.</span></footer>
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
      <p className="panel-sub rv" style={iv(4)}><strong>We also tried Mondrian CP:</strong> separate calibration cutoffs for each class. In the earlier six-class Architecture A experiment at CP α = 0.10, coverage averaged 92–96% for every class across 100 family splits, but only 20% of answers contained a single class. These are averages across splits, not a guarantee for every class on each split or current Architecture B results.</p>
      <p className="panel-sub rv" style={iv(5)}><strong>Why LAC:</strong> in that same experiment, global LAC produced smaller sets and more single-class answers than global APS and RAPS: 1.67 classes per set and 64.5% single answers, versus 2.00 and 23.5% for both alternatives, averaged across 100 family splits at α = 0.10. RAPS selected a zero regularization penalty, so it matched APS. This supports LAC's simpler visor answers; it is not evidence that LAC is always better.</p>
    </div>
  );
}

function SetSizes() {
  const rv = useReveal<HTMLDivElement>(0.8);
  const cutoff = cpComparison.variants.family_weighted.probability_cutoff;
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
      <p className="unc-results-note rv" style={iv(3)}>Observed coverage, not a guarantee for each size. The size-6 result comes from only {int(U.by_set_size.find((row) => row.size === 6)?.n ?? 0)} crops. Mathematically, LAC can return an empty set if every class probability is below the inclusion cutoff. {cutoff <= 1 / U.by_class.length && <>At our current cutoff of {pct(cutoff, 2)}, this cannot happen: the {U.by_class.length} probabilities sum to 100%, so at least one must be {pct(1 / U.by_class.length, 2)} or higher. </>}The implementation also keeps the top class if a different cutoff would produce an empty set. OOD rejection means re-aim, not an empty CP prediction; a no-defect-only set is also not empty.</p>
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

function GateCombination() {
  const rv = useReveal<HTMLDivElement>(0.8);
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>Why combine the gates with Bonferroni?</h2>
      <p className="panel-sub rv" style={iv(1)}>Two checks give two opportunities to reject a valid crop. Bonferroni accounts for both: we double the smaller of the AoN and kNN p-values, capped at 1. The combined cutoff is <strong>{oodComparison.alpha_ood}</strong>. We reject at or below it, which is equivalent to rejecting if <strong>either individual p-value is at or below {oodComparison.alpha_ood / 2}</strong>. Each detector does not get the full combined cutoff.</p>
      <ol className="unc-steps rv" style={iv(2)}>
        <li><b>Combined p = min(1, 2 × min(p_AoN, p_kNN)).</b> Reject when combined p ≤ {oodComparison.alpha_ood}; otherwise pass the crop to the classification decision.</li>
        <li><b>Example: 0.02 and 0.20.</b> Combined p = 0.04, so reject at the current 0.05 cutoff.</li>
        <li><b>Example: 0.03 and 0.04.</b> Combined p = 0.06, so accept at 0.05. Testing each separately at 0.05 would apply a different rule.</li>
      </ol>
      <div className="unc-table-wrap rv" style={iv(3)}>
        <table className="unc-size-table">
          <caption>Historical gate experiment · OOD α = 0.08 (current: 0.05) · FRR means and 5th–95th percentile ranges over 200 halvings of 88 calibration families</caption>
          <thead><tr><th scope="col">Rule tested</th><th scope="col">Defect FRR</th><th scope="col">No-defect FRR</th><th scope="col">Non-aircraft inputs shown as defect / possible</th></tr></thead>
          <tbody>{experiments.gate_combinations.map((r, i) => <tr key={r.name} className={i === 0 ? "unc-ood-selected" : undefined}><th scope="row">{r.name}</th><td>{r.defect_frr}</td><td>{r.clean_frr}</td><td>{r.offtarget_shown}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="panel-sub rv" style={iv(4)}>Simes also lets two moderately small p-values trigger rejection together. For two detectors its combined value is min(2 × the smaller p-value, the larger p-value); the 0.03 / 0.04 example would be rejected at 0.05. In the historical 0.08 experiment below, Simes reduced non-aircraft defect displays from 3.2% to 2.8%, but increased false rejection of defects from 5.0% to 5.6% and clean crops from 3.6% to 4.2%. AoN alone rejected still more valid crops. This supported retaining Bonferroni to prioritize low false rejection. These historical rates have not been relabeled as measurements at 0.05.</p>
      <p className="unc-results-note rv" style={iv(5)}>Bonferroni does not require the two detectors to be independent, provided their individual p-values are valid. Simes needs additional dependence conditions that we have not established here. Both detectors share ResNet features, so independence should not be assumed. This is a reason for the combination rule, not proof that our full pipeline meets every conformal assumption; the nominal cutoff is not a claim that observed FRR equals it.</p>
      <p className="unc-results-note rv" style={iv(6)}>These calibration-stage numbers are separate from the held-out OOD comparison below. “Shown” counts a defect or POSSIBLE message, whereas FAR below counts every crop passing the gate, including silent outputs. No AUPRC was recorded for this Simes experiment; the measured AUPRC results below belong to the separate detector comparison.</p>
    </div>
  );
}

function OodThreshold() {
  const rv = useReveal<HTMLDivElement>(0.8);
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>Why OOD α = {oodComparison.alpha_ood}?</h2>
      <p className="panel-sub rv" style={iv(1)}>The p-value is calculated for each crop; α is the rejection cutoff we choose. Increasing α rejects more crops, reducing false acceptance but increasing false rejection. We compared five cutoffs while keeping the model and Bonferroni rule fixed.</p>
      <div className="unc-table-wrap rv" style={iv(2)}>
        <table className="unc-size-table">
          <caption>Calibration-stage sweep · 200 family halvings · FRR means and 5th–95th percentile ranges · external passage rates use all 88 calibration families</caption>
          <thead><tr><th scope="col">Combined α</th><th scope="col">Defect FRR</th><th scope="col">No-defect FRR</th><th scope="col">Seen non-aircraft passage</th><th scope="col">Wallpaper passage</th><th scope="col">Static passage</th></tr></thead>
          <tbody>{experiments.ood_alpha.map((r) => <tr key={r.alpha} className={Number(r.alpha) === oodComparison.alpha_ood ? "unc-ood-selected" : undefined}><th scope="row">{r.alpha}</th><td>{r.defect_frr}</td><td>{r.clean_frr}</td><td>{r.seen_far}</td><td>{r.wallpaper_far}</td><td>{r.static_far}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="panel-sub rv" style={iv(3)}>At 0.05 fewer valid crops are rejected, but more non-aircraft inputs pass. At 0.10 and above rejection of valid crops rises further. We changed from 0.08 to 0.05 to prioritize fewer false rejections, accepting that more non-aircraft inputs pass. This is a chosen operating point, not a proven universal optimum. The non-aircraft-photo column contains training-exposed examples and cannot establish generalization to unseen outliers; wallpaper and static are separate stress tests.</p>
      <p className="unc-results-note rv" style={iv(4)}>A principled selection needs a requirement set in advance. To prioritize low FRR, specify the maximum acceptable FAR, then choose the smallest α meeting that constraint on independent validation data, accounting for uncertainty and family grouping. Lock the threshold before final testing. Minimizing FRR alone would favor never rejecting anything. AUPRC summarizes score ranking across thresholds, so it does not select α by itself. The deployed α is {oodComparison.alpha_ood}, equivalent to {oodComparison.alpha_ood / 2} per detector. CP α remains 0.10.</p>
    </div>
  );
}

function OodComparison() {
  const rv = useReveal<HTMLDivElement>(0.8);
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>Why we keep ResNet + AoN + kNN</h2>
      <p className="panel-sub rv" style={iv(1)}>We prioritize a low false rejection rate (FRR) above reducing false acceptance further: usable aircraft crops should reach the inspection model. At the current 0.05 cutoff, ResNet + AoN + kNN has the lowest D1 FRR in this comparison. ResNet + AoN alone has lower D5 FRR, so no configuration wins on every measure. We retain the combined gate and its ResNet features. We accept the higher held-out false acceptance rate (FAR) shown in this comparison.</p>
      <div className="unc-table-wrap rv" style={iv(2)}>
        <table className="unc-size-table">
          <caption>Architecture B · quality gate off · OOD α = {oodComparison.alpha_ood} · sorted by D1 FRR, lowest first · green marks our chosen configuration</caption>
          <thead><tr><th scope="col">OOD configuration</th><th scope="col">FAR ↓</th><th scope="col">D1 FRR ↓</th><th scope="col">D5 FRR ↓</th><th scope="col">Gate AUPRC ↑</th><th scope="col">Raw-score AUPRC ↑</th></tr></thead>
          <tbody>{[...oodComparison.variants].sort((a, b) => a.d1_frr - b.d1_frr).map((row) => <tr key={row.name} className={row.selected ? "unc-ood-selected" : undefined}>
            <th scope="row">{row.name}</th>
            <td>{pct(row.far, 2)}</td><td>{pct(row.d1_frr, 2)}</td><td>{pct(row.d5_frr, 2)}</td><td>{pct(row.gate_auprc, 2)}</td><td>{row.raw_auprc === null ? "Not applicable" : pct(row.raw_auprc, 2)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <p className="unc-results-note rv" style={iv(3)}>FRR: valid aircraft crops stopped by the gate. FAR: non-aircraft crops that pass it, regardless of what the visor later displays. Evaluated on {int(oodComparison.test_crops)} D1 crops, {int(oodComparison.d5_crops)} D5 crops and {int(oodComparison.offtarget_crops)} held-out non-aircraft crops. Rates give each crop equal weight; no thresholds were tuned on these test results.</p>
      <p className="panel-sub rv" style={iv(3)}><strong>D5 external test: 0.3% FRR.</strong> The deployed gate rejected 3 of 1,008 evaluated crops from Aircraft Defect Detection, the University of Technology Sydney workspace dataset on Roboflow (version 3). This is our external evaluation subset, not the full dataset; nothing was fitted or calibrated on D5.</p>
      <p className="panel-sub rv" style={iv(4)}>AoN and DINOv2 have different jobs. AoN learns to distinguish aircraft crops from non-aircraft examples; DINOv2 supplies image features and still needs a detector such as AoN or kNN. Our AoN uses features already produced by ResNet, avoiding a second image encoder. Combining it with kNN adds a learned non-aircraft check to the distance check: ResNet kNN alone accepted {pct(oodComparison.variants[1].far, 2)} of non-aircraft crops, compared with {pct(oodComparison.variants[2].far, 2)} for the combined gate.</p>
      <p className="unc-results-note rv" style={iv(5)}>Controlled benchmark: non-aircraft evaluation categories were excluded from fitting. These are experimental comparison results at the current OOD cutoff, not direct measurements of deployed performance. DINOv2 + AoN achieved the lowest FAR, but rejected more valid D1 and D5 crops.</p>
      <p className="panel-sub rv" style={iv(6)}><strong>What AUPRC tells us.</strong> Area under the precision–recall curve summarizes how well the detector finds non-aircraft crops across different rejection thresholds. Precision asks: of the crops flagged as non-aircraft, how many really are? Recall asks: of all non-aircraft crops, how many did we find? Higher AUPRC means a better balance across thresholds; it does not mean that this percentage of predictions is correct, or guarantee low FRR at our chosen threshold.</p>
      <p className="unc-results-note rv" style={iv(7)}>We calculate AUPRC using average precision, with non-aircraft crops as the positive class. Gate AUPRC uses the calibrated p-value ranking; raw-score AUPRC uses the detector before conformal calibration. Calibration can tie scores, so the two can differ. Combined gates have no single raw detector score.</p>
      <p className="unc-results-note rv" style={iv(8)}>AUPRC depends on how common outliers are. Our pooled benchmark has {int(oodComparison.offtarget_crops)} non-aircraft and {int(oodComparison.test_crops)} D1 crops. Reweighting to the 1:5 outlier-to-normal ratio used in <a href="https://arxiv.org/abs/1812.04606" target="_blank" rel="noreferrer">Deep Anomaly Detection with Outlier Exposure</a> gives {pct(oodComparison.variants.find((row) => row.selected)!.gate_auprc_1_to_5, 2)} for our combined gate. Different datasets and training procedures still prevent a direct performance comparison with that paper.</p>
    </div>
  );
}

function CalibratorExperiments({ rep }: { rep: Report }) {
  const rv = useReveal<HTMLDivElement>(0.8);
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>Calibration alternatives: why keep one temperature?</h2>
      <p className="panel-sub rv" style={iv(1)}>Temperature scaling is one shared confidence dial and preserves the winning class. Vector scaling learns a dial for each class; vector+bias also learns class offsets. Isotonic calibration learns a separate correction chart for each class, then normalizes the probabilities. These more flexible methods can change class rankings and fit quirks of the calibration data.</p>
      <div className="unc-table-wrap rv" style={iv(2)}>
        <table className="unc-size-table">
          <caption>Historical calibrator experiment (OOD α = 0.08 for CP columns) · fitted on 4,818 validation crops · calibration metrics evaluated on 2,116 cal-half crops, with equal family weight</caption>
          <thead><tr><th scope="col">Calibrator</th><th scope="col">NLL ↓</th><th scope="col">ECE ↓</th><th scope="col">ECE change vs T (95% interval)</th><th scope="col">CP family coverage</th><th scope="col">Mean set size</th></tr></thead>
          <tbody>{experiments.calibrators.map((r) => <tr key={r.name}><th scope="row">{r.name}</th><td>{r.nll}</td><td>{r.ece}</td><td>{r.ece_delta_ci}</td><td>{r.family_coverage}</td><td>{r.set_size}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="panel-sub rv" style={iv(3)}>Isotonic achieved a lower measured ECE than temperature scaling (0.0371 vs 0.0460), but the family-bootstrap interval for the improvement includes zero. We kept the simpler, rank-preserving temperature correction because these experiments did not establish a clear, reliable benefit from the added flexibility. This does not mean isotonic performed worse.</p>
      <p className="unc-results-note rv" style={iv(4)}>NLL penalizes assigning low probability to the true class. ECE measures the average gap between confidence and accuracy; 0.0460 means 4.60 percentage points. These are cal-half, family-weighted results, separate from the main held-out test ECE of {rep.arch_a.ece.raw.toFixed(5)} before scaling and {rep.arch_a.ece.tempered.toFixed(5)} after. CP columns average 200 splits of the cal-half families at α = 0.10. The 95% ECE intervals resample families. AUPRC measures OOD detection, so it is not the metric for this confidence-calibration comparison.</p>
    </div>
  );
}

function MissGuardExperiment() {
  const rv = useReveal<HTMLDivElement>(0.8);
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>Defect-miss guard: tested, left disabled</h2>
      <p className="panel-sub rv" style={iv(1)}>This optional one-sided conformal test only permits silence when the crop's no-defect probability is unusually high compared with real defect calibration crops. If the test fails, a lone no-defect answer becomes “POSSIBLE …: check”. It is an additional check on silence, separate from the OOD gate.</p>
      <div className="unc-table-wrap rv" style={iv(2)}>
        <table className="unc-size-table">
          <caption>Historical guard experiment · OOD α = 0.08 (current: 0.05) · CP α = 0.10 · means and 5th–95th percentile ranges across family halvings</caption>
          <thead><tr><th scope="col">Guard setting</th><th scope="col">Defects shown as nothing</th><th scope="col">No-defect crops shown as nothing</th></tr></thead>
          <tbody>{experiments.guard.map((r) => <tr key={r.name}><th scope="row">{r.name}</th><td>{r.silent_miss}</td><td>{r.clean_silence}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="panel-sub rv" style={iv(3)}>At α_miss = 0.05–0.10 the guard changed no decisions. At 0.02 it blocked all silence: with at most 44 calibration families, the smallest possible p-value is at least 1/45, about 2.22%, so none can pass a 2% cutoff. Zero silent misses here came from never allowing silence, not from better defect recognition.</p>
      <p className="unc-results-note rv" style={iv(4)}>The guard remains available but disabled. More calibration families could make smaller p-values possible; that alone does not guarantee a useful improvement. These local results are separate from the main test-set silent-miss rate.</p>
    </div>
  );
}

function CalibrationSplitExperiment() {
  const rv = useReveal<HTMLDivElement>(0.8);
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={iv(0)}>Would a larger calibration set help?</h2>
      <p className="panel-sub rv" style={iv(1)}>We tested reserving 20% of training families for calibration, leaving 80% to train a new model. That supplied 166 calibration families, or 210 when combined with 44 cal-half families. The all-train reference here is the gc7_fix retrain, not the deployed gc7 model. Evaluation used 44 separate cal-half families, repeated over 200 halvings.</p>
      <div className="unc-table-wrap rv" style={iv(2)}>
        <table className="unc-size-table">
          <caption>Historical split experiment · OOD α = 0.08 (current: 0.05) · CP α = 0.10 · means with 5th–95th percentile ranges across halvings</caption>
          <thead><tr><th scope="col">Setup</th><th scope="col">Cal families</th><th scope="col">Family coverage</th><th scope="col">Set size</th><th scope="col">Silent miss</th><th scope="col">Clean false alarm</th><th scope="col">Family FRR</th></tr></thead>
          <tbody>{experiments.calibration_split.map((r) => <tr key={r.name}><th scope="row">{r.name}</th><td>{r.families}</td><td>{r.coverage}</td><td>{r.set_size}</td><td>{r.silent_miss}</td><td>{r.false_alarm}</td><td>{r.false_reject}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="panel-sub rv" style={iv(3)}>With 166 calibration families, the observed coverage range narrowed from 85.4–96.6% to 91.5–95.2%, and clean-crop false alarms decreased. But silent misses rose from 0.4% to 1.0%, and false rejection also increased. The experiment changed the training split and model as well as calibration size, so it cannot isolate the effect of adding calibration data.</p>
      <p className="unc-results-note rv" style={iv(4)}>Coverage and FRR give each family equal weight; set size, silent misses and clean false alarms give each crop equal weight. A clean false alarm means any visor message, including re-aim. These exploratory results do not replace the deployed model or its held-out test results. All three experiments above come from saved calibration-stage reports; the main test figures are reported separately.</p>
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
      <p className="panel-sub rv" style={iv(1)}>It catches what the model was never meant to see. The deployed AoN and kNN gate together wrongly stop {pct(s.actions["re-aim"])} of normal test crops. AoN supplies a learned non-aircraft score alongside the kNN distance check.</p>
      <ol className="unc-steps rv" style={iv(2)}>
        <li><b>Features.</b> The model turns every crop into 768 numbers (its 8-view features).</li>
        <li><b>A learned line.</b> A logistic regression weighs those numbers: {int(U.aon.n_pos)} training crops of aircraft skin on one side, {int(U.aon.n_neg)} crops of people, tools, vehicles and scenes on the other (outlier exposure). Its score is the weighted sum plus a bias; past 0 it leans "not aircraft".</li>
        <li><b>A conformal p-value.</b> The score is compared with the scores of the 88 calibration photo groups: how unusual is this crop for aircraft skin? With the kNN distance's p-value, the smaller one doubled must stay above {rep.arch_a.alpha_ood}, or the visor says re-aim. With both detectors, that means each individual p-value must be above {rep.arch_a.alpha_ood / 2}.</li>
      </ol>
      <p className="panel-sub rv" style={iv(3)}><strong>Training:</strong> both D1 aircraft crops and labeled non-aircraft crops pass through the same fixed ResNet18 used by our defect model. Non-aircraft sources include Open Images V7 and Places365. Its layers 3 and 4 produce 768 feature values per crop. AoN learns from those features and their aircraft/non-aircraft labels; ResNet's weights stay fixed.</p>
      <p className="panel-sub rv" style={iv(4)}><strong>For an incoming crop:</strong> the same ResNet extracts its features. AoN standardizes them, forms a weighted sum plus a bias, and applies a sigmoid to produce a non-aircraft probability. This is its own binary logistic-regression output, not the MLP's softmax or the hybrid class probabilities. The score is compared with aircraft calibration scores to obtain an OOD p-value, then combined with the separate kNN OOD p-value for rejection.</p>
      <figure className="unc-false-reject rv" style={iv(5)}>
        <img src={falseRejection.image} width={224} height={224} loading="lazy" alt="D1 corrosion crop from family 533, incorrectly rejected by the aircraft-or-not check" />
        <figcaption><strong>A false rejection: family {falseRejection.family}.</strong> This D1 test crop is labelled corrosion, but the visor says re-aim. AoN's p-value is {falseRejection.p_aon.toFixed(4)}, below its {falseRejection.per_detector_cutoff} cutoff; kNN's is {falseRejection.p_knn.toFixed(4)}, which passes. In total, {falseRejection.rejected_crops} test crops from this family are rejected, all by AoN alone. The rejection does not mean the crop has no defect.</figcaption>
      </figure>
      <div className="rv" style={iv(5)}><AonExplorer d={U.aon} shown={rv.shown} /></div>
    </div>
  );
}
