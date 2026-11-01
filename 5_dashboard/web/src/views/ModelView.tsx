import type React from "react";
import { motion } from "motion/react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { cropUrl, getReport, type AlphaResult, type NoDefect, type Report } from "../api/client";
import { AnimatedNumber } from "../components/AnimatedNumber";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { RecallDumbbell } from "../components/charts/RecallDumbbell";
import { ReliabilityChart } from "../components/charts/ReliabilityChart";
import { HelmetTitle } from "../components/HelmetTitle";
import { RuleDraw } from "../components/RuleDraw";
import { ANATOMY_STEPS, ModelAnatomy } from "../components/ModelAnatomy";
import { SQUARE_AT, SquareScene } from "../components/SquareScene";
import { StateGlyph } from "../components/StateBadge";
import { useReveal, type Reveal } from "../lib/useReveal";
import { cap, className, int, pct } from "../lib/format";
import { usePresent } from "../state/present";
import "../styles/model.css";

const ease = [0.22, 1, 0.36, 1] as const;

export function ModelView() {
  const [rep, setRep] = useState<Report | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { getReport().then(setRep).catch((e: Error) => setErr(e.message)); }, []);
  const present = usePresent();

  if (err) return <div className="model"><HelmetTitle /><section className="model-empty"><h2 className="view-title">Model</h2><p>{err}</p></section></div>;
  if (!rep) return <div className="model"><HelmetTitle /><section className="model-empty"><h2 className="view-title">Model</h2><p className="muted">Computing the report… On a new computer the first start takes about 2 minutes (the model reads every validation and test crop once).</p></section></div>;

  // each section is one presentation step; "How it works" and the square scene are several, one per stop
  const sections: [string, ReactNode][] = [
    ["Headline", <Headline rep={rep} />],
    ["Anatomy", null],
    ["Classes", <Classes rep={rep} />],
    ...(rep.no_defect ? [["NoDefect", <NoDefectSection rep={rep} nd={rep.no_defect} />] as [string, ReactNode]] : []),
    ["Square", null],
    ["Evidence", <Evidence rep={rep} />],
  ];
  const sizes = sections.map(([name]) => (name === "Square" ? SQUARE_AT.length : name === "Anatomy" ? ANATOMY_STEPS : 1));
  // step 0 is the title slide; the sections follow
  const firsts = sizes.map((_, i) => 1 + sizes.slice(0, i).reduce((a, b) => a + b, 0));
  const total = 1 + sizes.reduce((a, b) => a + b, 0);
  const dimAt = (step: number) => present.on && present.step !== step;
  return (
    <div className="model">
      <HelmetTitle step={0} />
      {sections.map(([name, node], i) => (name === "Square" ? <section key={name} className="model-section"><SquareScene first={firsts[i]} dimAt={dimAt} /></section>
        : name === "Anatomy" ? <section key={name} className="model-section"><ModelAnatomy first={firsts[i]} dimAt={dimAt} /></section>
        : <motion.section key={name} data-step={firsts[i]} className="model-section" animate={{ opacity: dimAt(firsts[i]) ? 0.25 : 1 }} transition={{ duration: 0.3 }}>
          {node}
        </motion.section>
      ))}
      {present.on && <p className="step-hint">Section {present.step + 1} of {total} · ← → to move</p>}
      <Provenance rep={rep} />
    </div>
  );
}

function Headline({ rep }: { rep: Report }) {
  const t = rep.test;
  const weak = [...t.per_class].sort((a, b) => a.recall - b.recall).slice(0, 2).map((c) => className(c.cls).toLowerCase());
  const nd = rep.no_defect, base = rep.baseline;
  const rv = useReveal<HTMLDivElement>(0.85);
  return (
    <div className="headline" ref={rv.ref} data-shown={rv.shown}>
      <div>
        <h2 className="view-title rv" style={{ "--i": 0 } as React.CSSProperties}>{rep.label}, the main model</h2>
        <p className="view-sub rv" style={{ "--i": 1 } as React.CSSProperties}>{rep.description}. Trained on {rep.trained_on}.</p>
        <motion.p className="headline-figure" initial={false}
          animate={{ clipPath: rv.shown ? "inset(0 0% 0 0)" : "inset(0 100% 0 0)", y: rv.shown ? 0 : 18 }}
          transition={{ duration: rv.t(0.9), ease }}><AnimatedNumber value={(nd ? nd.defects_best.accuracy : t.accuracy) * 100} format={(v) => `${v.toFixed(1)}%`} /></motion.p>
        <p className="headline-sentence rv" style={{ "--i": 3 } as React.CSSProperties}>{nd ? <>right defect class on {int(nd.n_defects)} unseen defect crops from {t.groups} separate photo groups.</> : <>top class correct on {int(t.n)} unseen test crops from {t.groups} separate photo groups.</>}</p>
        <p className="headline-finding rv" style={{ "--i": 4 } as React.CSSProperties}>{`${cap(weak[0])} and ${weak[1]} are the weak spots.`}</p>
      </div>
      <dl className="headline-facts">
        <div className="rv" style={{ "--i": 2 } as React.CSSProperties}><dt>Macro-F1, every class counting equally{nd ? ` (all ${rep.classes.length})` : ""}</dt><dd><span className="fig big">{t.macro_f1.toFixed(3)}</span>{t.ci && <span className="sub"> 95% interval {t.ci[0].toFixed(3)}–{t.ci[1].toFixed(3)}{nd ? ` · balanced accuracy ${pct(t.balanced_accuracy)}` : ""}</span>}</dd></div>
        <div className="rv" style={{ "--i": 3 } as React.CSSProperties}><dt>What the inspector sees, through architecture B</dt><dd>The answer holds the true class <span className="fig big">{pct(served(rep).cp_coverage)}</span><span className="sub"> of the time (target {pct(1 - rep.arch_a.alpha, 0)}); a single wrong class with no warning on {pct(served(rep).single_class_wrong)}, against {pct(1 - served(rep).accuracy)} if it always showed its top class</span></dd></div>
        {!nd && <div className="rv" style={{ "--i": 3 } as React.CSSProperties}><dt>Balanced accuracy</dt><dd><span className="fig big">{pct(t.balanced_accuracy)}</span></dd></div>}
        {nd && <div className="rv" style={{ "--i": 3 } as React.CSSProperties}><dt>Intact surface it recognises</dt><dd><span className="fig big">{pct(nd.recall)}</span><span className="sub"> of {int(nd.n)} no-defect test crops; {pct(nd.defects_called_clear)} of defects called no defect. The 6-class model called every clean crop a defect.</span></dd></div>}
        {nd && base && <div className="rv" style={{ "--i": 4 } as React.CSSProperties}><dt>Against {base.label}, on the same defect crops</dt><dd>Accuracy <span className="fig">{nd.defects_best.accuracy.toFixed(3)}</span> · macro-F1 <span className="fig">{nd.defects_best.macro_f1.toFixed(3)}</span><span className="sub"> (v2: {base.accuracy.toFixed(3)} · {base.macro_f1.toFixed(3)}). The 7th class cost nothing on defects. Over all {int(t.n)} test crops, no defect included, its top answer is right {pct(t.accuracy)} of the time.</span></dd></div>}
        {!nd && base && <div className="rv" style={{ "--i": 4 } as React.CSSProperties}><dt>Against {base.label}, same split</dt><dd><span className="fig">{pct(base.accuracy)}</span> correct · macro-F1 <span className="fig">{base.macro_f1.toFixed(3)}</span></dd></div>}
        {rep.d5 && <div className="rv" style={{ "--i": 5 } as React.CSSProperties}><dt>Photos from another dataset (D5, not copies of D1)</dt><dd><span className="fig">{pct(rep.d5.accuracy)}</span> correct on {int(rep.d5.n)} crops <span className="sub">({Object.entries(rep.d5.counts).map(([c, n]) => `${int(n)} ${className(c).toLowerCase()}`).join(", ")}; previous model {pct(rep.d5.base_accuracy)})</span></dd></div>}
      </dl>
    </div>
  );
}

function Classes({ rep }: { rep: Report }) {
  const [row, setRow] = useState<number | null>(null);
  const rows = rep.test.per_class.map((c) => ({ cls: c.cls, now: c.recall, before: rep.baseline?.recall[c.cls] ?? null }));
  const recall = useReveal<HTMLDivElement>(0.8), confusion = useReveal<HTMLDivElement>(0.8);
  return (
    <div className="pair">
      <div className="panel" ref={recall.ref} data-shown={recall.shown}>
        <RuleDraw />
        <h2 className="panel-title rv" style={{ "--i": 0 } as React.CSSProperties}>How often each class is caught</h2>
        <p className="panel-sub rv" style={{ "--i": 1 } as React.CSSProperties}>Recall on the test crops. Grey is {rep.baseline?.label ?? "the previous model"}, ink is this one.{rep.no_defect ? " The 6-class model had no no-defect class, so it caught none." : ""}</p>
        <RecallDumbbell rows={rows} reveal={recall} />
        <p className="chart-key rv" style={{ "--i": 2 } as React.CSSProperties}><span className="key-dot key-before" /> {rep.no_defect ? "6-class model (v2)" : "Previous model (same split)"} <span className="key-dot key-now" /> {rep.label} · the small figure is the change in points</p>
      </div>
      <div className="panel" ref={confusion.ref} data-shown={confusion.shown}>
        <RuleDraw />
        <h2 className="panel-title rv" style={{ "--i": 0 } as React.CSSProperties}>What it confuses</h2>
        <p className="panel-sub rv" style={{ "--i": 1 } as React.CSSProperties}>Rows are the true class, columns what the model called it, as a percentage of the row. Pick a row to focus it.</p>
        <ConfusionMatrix classes={rep.classes} matrix={rep.test.confusion} selected={row} onSelect={setRow} label="Test confusion matrix, row percentages" reveal={confusion} />
      </div>
    </div>
  );
}

/** What the 7th class adds, as a 2 x 2 sheet: clean crops recognised | defects called no defect, then the LaMa
 *  check | the validation-chosen cut. Each cell is a heading, a line and four to six bars or rows. */
function NoDefectSection({ rep, nd }: { rep: Report; nd: NoDefect }) {
  const rv = useReveal<HTMLDivElement>(0.8);
  const recog: Bar[] = [
    { label: "Healed: damage painted out", n: nd.n_by_source.healed, v: nd.recall_by_source.healed },
    { label: "Clean: unlabelled areas", n: nd.n_by_source.clean_cropper, v: nd.recall_by_source.clean_cropper },
    ...(nd.extra ? [{ label: "Spare crops, never trained on", n: nd.extra.n, v: nd.extra.recall }] : []),
    { label: "6-class model, same crops", n: nd.n, v: 0, muted: true },
  ];
  const missed: Bar[] = Object.entries(nd.defects_called_clear_by_class).sort((a, b) => b[1] - a[1])
    .map(([c, v]) => ({ label: className(c), v, scale: 0.1 }));
  const dec = nd.decoys;
  const lama: Bar[] = dec ? [
    { label: "Painted patch: called no defect", v: dec.called_no_defect.decoy, scale: 0.1, tone: "miss" },
    { label: "Untouched: called no defect", v: dec.called_no_defect.original, scale: 0.1, tone: "miss", muted: true },
    { label: "Painted patch: right class", v: dec.right_class.decoy },
    { label: "Untouched: right class", v: dec.right_class.original, muted: true },
  ] : [];
  const served = rep.arch_a.by_alpha[rep.arch_a.alpha.toFixed(2)];
  return (
    <div className="nd" ref={rv.ref} data-shown={rv.shown}>
      <div className="panel nd-head">
        <RuleDraw />
        <h2 className="panel-title rv" style={{ "--i": 0 } as React.CSSProperties}>A 7th class: no defect</h2>
        <p className="panel-sub rv" style={{ "--i": 1 } as React.CSSProperties}>D1 has no photos of undamaged skin, so the 6-class model named a defect on every clean panel. We made {int(nd.train.train + nd.train.val + nd.train.test)} no-defect crops from D1&rsquo;s own photos (the Data page shows how) and retrained. What that bought, and what it costs:</p>
      </div>
      <div className="nd-grid">
        <NdCell i={2} rv={rv} title="Intact surface it recognises" sub={`Share of the ${int(nd.n)} no-defect test crops it calls no defect, by where they came from.`} bars={recog} tone="clear" />
        <NdCell i={2} rv={rv} title="Defects it calls no defect" sub={`${pct(nd.defects_called_clear)} of the ${int(nd.n_defects)} defect test crops, mostly faint scratches and cracks. Bars run to 10%.`} bars={missed} tone="miss" />
        {dec && <NdCell i={3} rv={rv} title="Is it just spotting LaMa's paint?" sub={`We painted a clean patch into ${int(dec.n)} defect crops. Barely more are called no defect, and the class stays right: it did not learn "painted = intact".`} bars={lama} tone="clear" />}
        {nd.dial && (
          <div className="nd-cell rv" style={{ "--i": 3 } as React.CSSProperties}>
            <h3 className="nd-h3">Fewer misses, chosen on validation</h3>
            <p className="nd-sub">Say no defect only when p(no defect) passes a cut set on validation.{served?.defect_called_clear != null ? <> Through architecture B, only {pct(served.defect_called_clear)} of defects are shown as <b>No defect</b> (nothing on the visor).</> : null}</p>
            <table className="nd-dial">
              <thead><tr><th scope="col">Cut set to miss, val</th><th scope="col">Defects missed, test</th><th scope="col">Clean crops caught</th></tr></thead>
              <tbody>
                {nd.dial.map((d) => <tr key={d.val_miss_target}><th scope="row">{pct(d.val_miss_target, 0)}</th><td className="fig">{pct(d.test_defects_missed)}</td><td className="fig">{pct(d.test_no_defect_recall, 0)}</td></tr>)}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

interface Bar { label: string; v: number; n?: number; scale?: number; tone?: "clear" | "miss"; muted?: boolean }

/** One cell of the no-defect sheet: a heading, a line, and bars (each scaled to `scale`, 1 by default). */
function NdCell({ i, rv, title, sub, bars, tone }: { i: number; rv: Reveal; title: string; sub: string; bars: Bar[]; tone: "clear" | "miss" }) {
  return (
    <div className="nd-cell rv" style={{ "--i": i } as React.CSSProperties}>
      <h3 className="nd-h3">{title}</h3>
      <p className="nd-sub">{sub}</p>
      <ol className="gate-bars nd-bars">
        {bars.map((x, k) => (
          <li key={x.label} className={x.muted ? "is-muted" : undefined}>
            <span>{x.label}{x.n != null && <small className="nd-n">{int(x.n)}</small>}</span>
            <span className="gate-track"><motion.span className={`gate-fill nd-${x.tone ?? tone}`} initial={false}
              animate={{ scaleX: rv.shown ? Math.min(1, x.v / (x.scale ?? 1)) : 0 }} transition={{ duration: rv.t(0.7), delay: rv.t(0.06) * k + rv.t(0.2), ease }} /></span>
            <span className="fig">{pct(x.v, x.scale ? 1 : 0)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Architecture B at the alpha the dashboard serves. */
const served = (rep: Report) => rep.arch_a.by_alpha[rep.arch_a.alpha.toFixed(2)];

type MixState = "detected" | "clear" | "ambiguous" | "unclear" | "retake" | "cannot_assess";
/** The answer types, in the order the bar stacks them. With a no-defect class, a single answer is split into a
 *  defect and "no defect". */
function mixOf(r: AlphaResult, nd: boolean): [MixState, string, number][] {
  const nothing = r.actions["nothing shown"] ?? 0, possible = r.actions["possible defect"] ?? 0;
  const clear = nd ? r.clear ?? 0 : 0, one = r.actions["show class"] - Math.max(0, clear - nothing);
  return ([
    ["detected", nd ? "One defect" : "One class", one], ...(nd ? [["clear", "No defect", clear]] : []),
    ["ambiguous", nd ? "Ambiguous or possible" : "Ambiguous", r.actions.ambiguous + possible], ["unclear", "Unclear", r.actions.unclear],
    ["retake", "Bad image", r.actions["re-take"]], ["cannot_assess", "Cannot assess", r.actions["re-aim"]],
  ] as [MixState, string, number][]).filter(([, , v]) => v > 0);
}

export function Confidence({ rep }: { rep: Report }) {
  const a = rep.arch_a;
  const alphas = Object.keys(a.by_alpha).sort();
  const [key, setKey] = useState(a.alpha.toFixed(2));
  const r = a.by_alpha[key];
  const honest = useReveal<HTMLDivElement>(0.8), sets = useReveal<HTMLDivElement>(0.8);
  const raw = a.reliability.raw.find((b) => b.lo === 0.9);
  const nd = !!rep.no_defect, mix = mixOf(r, nd);
  return (
    <div className="pair">
      <div className="panel" ref={honest.ref} data-shown={honest.shown}>
        <RuleDraw />
        <h2 className="panel-title rv" style={{ "--i": 0 } as React.CSSProperties}>Are its percentages honest?</h2>
        <p className="panel-sub rv" style={{ "--i": 1 } as React.CSSProperties}>
          {raw && <>Raw, when the model said over 90% ({pct(raw.confidence, 0)} on average) it was right {pct(raw.accuracy, 0)} of the time. </>}
          Architecture B softens every percentage with one temperature (T = {a.temperature.toFixed(2)}, fitted by giving each validation photo family equal total weight). For this model it changes little: calibration error <span className="fig">{a.ece.raw.toFixed(3)}</span> → <span className="fig">{a.ece.tempered.toFixed(3)}</span>. These are crop-weighted test errors; temperature fitting gives each validation family equal weight. The top class never changes. A calibrated percentage is not a correctness guarantee for an individual crop.
        </p>
        <ReliabilityChart bins={a.reliability.tempered} before={a.reliability.raw} reveal={honest} label="Reliability on the test half: accuracy against stated confidence, before and after calibration" />
        <p className="chart-key rv" style={{ "--i": 2 } as React.CSSProperties}><span className="key-swatch key-bar" /> Calibrated <span className="key-raw" /> Raw, before calibration <span className="key-line key-dash" /> Perfectly honest · {int(a.test_half.n)} test crops, bands with 20+</p>
      </div>
      <div className="panel" ref={sets.ref} data-shown={sets.shown}>
        <RuleDraw />
        <h2 className="panel-title rv" style={{ "--i": 0 } as React.CSSProperties}>How many classes to show</h2>
        <p className="panel-sub rv" style={{ "--i": 1 } as React.CSSProperties}>Instead of always naming its top class, the model keeps every class whose calibrated percentage passes a cut-off set by conformal prediction. One class left is a <b>Defect detected</b>{nd ? <>, or <b>No defect</b> (nothing on the visor) when that class is no defect</> : null}; two or more is <b>Ambiguous</b>{nd ? <> (<b>Possible</b> when one of them is no defect)</> : null} and goes to a person. The cut-off is chosen so the answer holds the true class {pct(1 - a.alpha, 0)} of the time.</p>
        <div className="seg rv" style={{ "--i": 2 } as React.CSSProperties} role="radiogroup" aria-label="Error rate alpha">
          {alphas.map((k) => (
            <button key={k} role="radio" aria-checked={k === key} className="seg-btn" onClick={() => setKey(k)}>
              Aim for {pct(1 - Number(k), 0)}{Number(k) === a.alpha ? " · served" : ""}
            </button>
          ))}
        </div>
        <div className="mix rv" style={{ "--i": 3 } as React.CSSProperties} role="img" aria-label="Share of test crops given each kind of answer">
          {mix.map(([st, , v]) => (
            <motion.span key={st} className={`mix-seg mix-${st}`} initial={false} animate={{ flexGrow: sets.shown ? v : 0.0001 }}
              transition={{ duration: sets.t(0.7), ease }}>
              {v >= 0.08 && <span className="fig">{pct(v, 0)}</span>}
            </motion.span>
          ))}
        </div>
        <p className="chart-key rv" style={{ "--i": 3 } as React.CSSProperties}>
          {mix.map(([st, label, v]) => <span key={st} className="mix-key"><StateGlyph state={st} size={13} /> {label} {pct(v, 0)}</span>)}
        </p>
        <dl className="mix-facts rv" style={{ "--i": 4 } as React.CSSProperties}>
          <div><dt>Answer holds the true class</dt><dd><AnimatedNumber value={sets.shown ? r.cp_coverage * 100 : 0} format={(x) => `${x.toFixed(1)}%`} /></dd></div>
          <div><dt>Wrong with no warning</dt><dd><AnimatedNumber value={sets.shown ? r.single_class_wrong * 100 : 0} format={(x) => `${x.toFixed(1)}%`} /> <small>always naming the top class: {pct(1 - r.accuracy)}, at any α</small></dd></div>
          <div><dt>Needs a person</dt><dd><AnimatedNumber value={sets.shown ? r.human_attention * 100 : 0} format={(x) => `${x.toFixed(1)}%`} /></dd></div>
        </dl>
        <p className="coverage-ref rv" style={{ "--i": 5 } as React.CSSProperties}>Measured on the {int(a.test_half.n)} test crops ({a.test_half.families} photo groups) not used to set the cut-off. Aiming higher means bigger sets and more work for the inspector.</p>
      </div>
    </div>
  );
}

export function Refusals({ rep }: { rep: Report }) {
  const a = rep.arch_a, r = served(rep), g = a.gate_tests, q = a.alpha_quality != null;
  const rv = useReveal<HTMLDivElement>(0.8);
  const normal = r.actions["re-take"] + r.actions["re-aim"];
  const src = a.gates_by_source ?? [], clean = src.find((x) => x.source === "clean_cropper"), d1 = src.find((x) => x.source === "d1");
  const rows = [...(g?.sets ?? []).map((x) => ({ label: x.label, v: x.stopped, normal: false })),
    { label: "Normal test crops (false alarms)", v: normal, normal: true }];
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={{ "--i": 0 } as React.CSSProperties}>What it refuses to answer</h2>
      <div className="refuse">
        <div>
          <p className="panel-sub rv" style={{ "--i": 1 } as React.CSSProperties}>{q ? "Two checks run before any class is shown. Each compares the crop with normal photos the model was calibrated on." : "One check runs before any class is shown. It compares the crop with normal photos the model was calibrated on. Architecture B has no image-quality check: it flagged plain, clean panels as blurry."}</p>
          <ul className="legend rv" style={{ "--i": 2 } as React.CSSProperties}>
            {q && <li><StateGlyph state="retake" size={18} /><span><b>Image quality</b>: sharpness, brightness and blown-out or black pixels. Too unusual, and the answer is <b>Bad image: re-take</b>.</span></li>}
            <li><StateGlyph state="cannot_assess" size={18} /><span><b>Unfamiliar input</b>: how far the crop's features are from its 5 nearest training crops, and whether they look like aircraft skin at all (a check trained on photos of people, vehicles and scenes). Too far or not skin, and the answer is <b>Cannot assess: re-aim</b>. Without it the model names a defect on anything, confidently.</span></li>
          </ul>
          <p className="coverage-ref rv" style={{ "--i": 3 } as React.CSSProperties}>{q ? <>The price: they also stop {pct(normal, 0)} of normal test crops ({pct(r.actions["re-take"], 0)} quality, {pct(r.actions["re-aim"], 0)} unfamiliar). Blur mostly gets through.</> : <>The price: it also stops {pct(normal, 0)} of normal test crops{clean && d1 ? <> ({pct(d1.reaim, 0)} of defect crops, {pct(clean.reaim, 0)} of clean-cropper crops)</> : null}. Blurred, dark or glary photos are not stopped.</>}{q && clean && d1 ? <> Plain panels look blurry to the quality check: it sends {pct(clean.retake, 0)} of clean-cropper crops to re-take, against {pct(d1.retake, 0)} of defect crops.</> : null}</p>
        </div>
        <div className="rv" style={{ "--i": 2 } as React.CSSProperties}>
          <ol className="gate-bars" aria-label="Share of each kind of input stopped by the two checks">
            {rows.map((x, i) => (
              <li key={x.label} className={x.normal ? "is-normal" : undefined}>
                <span>{x.label}</span>
                <span className="gate-track"><motion.span className="gate-fill" initial={false} animate={{ scaleX: rv.shown ? x.v : 0 }}
                  transition={{ duration: rv.t(0.7), delay: rv.t(0.06) * i, ease }} /></span>
                <span className="fig">{pct(x.v, 0)}</span>
              </li>
            ))}
          </ol>
          {g && <p className="chart-key">{q ? <>Share stopped by the two checks, from the team's tests ({g.source})</> : <>Share not shown as a defect (stopped, or left silent), from the team's tests of architecture B ({g.source})</>}; the last row is measured here on {rep.label}.</p>}
        </div>
      </div>
    </div>
  );
}

function Evidence({ rep }: { rep: Report }) {
  const rv = useReveal<HTMLDivElement>(0.85);
  const errors = rep.arch_a.silent_errors;
  return (
    <div className="panel" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={{ "--i": 0 } as React.CSSProperties}>Mistakes it still shows without a warning</h2>
      <p className="panel-sub rv" style={{ "--i": 1 } as React.CSSProperties}>Test crops that passed the checks and got a single, wrong class, most confident first (at most two per photo group). These are the {pct(served(rep).single_class_wrong)} an inspector would not be warned about.{served(rep).defect_called_clear != null ? <> The worst kind, a defect shown as <b>No defect</b> (nothing on the visor), happens to {pct(served(rep).defect_called_clear ?? 0)} of defect crops.</> : null}</p>
      <div className="mistakes">
        {errors.map((e, i) => {
          const d = rv.t(0.07) * i;
          return (
            <motion.figure key={e.id} className="mistake" initial={false}
              animate={{ opacity: rv.shown ? 1 : 0, y: rv.shown ? 0 : 16 }} transition={{ duration: rv.t(0.45), delay: d, ease }}>
              <span className="mistake-frame">
                <img src={cropUrl(e.id)} alt={`True ${className(e.true)}, called ${className(e.pred)}`} loading="lazy" />
                <motion.span className="mistake-mark" aria-hidden="true" initial={false}
                  animate={{ opacity: rv.shown ? 1 : 0, scale: rv.shown ? 1 : 1.18 }} transition={{ duration: rv.t(0.3), delay: d + rv.t(0.35), ease }}>
                  <svg viewBox="0 0 100 100" preserveAspectRatio="none"><path d="M0,16 L0,0 L16,0 M84,0 L100,0 L100,16 M100,84 L100,100 L84,100 M16,100 L0,100 L0,84" /></svg>
                </motion.span>
              </span>
              <motion.figcaption initial={false} animate={{ opacity: rv.shown ? 1 : 0 }} transition={{ duration: rv.t(0.35), delay: d + rv.t(0.55) }}>
                <span>True <b>{className(e.true)}</b></span><span>Called {className(e.pred)} <span className="fig">{pct(e.confidence, 0)}</span></span>
              </motion.figcaption>
            </motion.figure>
          );
        })}
      </div>
    </div>
  );
}

function Provenance({ rep }: { rep: Report }) {
  return (
    <footer className="provenance">
      <span>Split: {rep.provenance.split} (validation {int(rep.val.n)} crops, test {int(rep.test.n)})</span>
      <span>Weights: {rep.provenance.weights.map((w) => `${w.file} (${w.modified})`).join(", ")}</span>
      <span>Architecture B: temperature fitted on validation, cut-off and check calibrated by the team on {int(rep.arch_a.cal.crops)} test crops ({rep.arch_a.cal.families} photo groups, {rep.arch_a.cal.created}), measured here on the other {int(rep.arch_a.test_half.n)} · report computed {rep.computed_at.slice(0, 10)}</span>
    </footer>
  );
}
