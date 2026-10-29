import type React from "react";
import { LayoutGroup, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import type { Report, StateId } from "../api/client";
import { pct } from "../lib/format";
import { useReveal } from "../lib/useReveal";
import { AnimatedNumber } from "./AnimatedNumber";
import { BUCKETS, PIPE, bucketLabel, cellKey, isOff } from "./PipelineScene";
import { RuleDraw } from "./RuleDraw";
import { StateGlyph } from "./StateBadge";

const ease = [0.22, 1, 0.36, 1] as const;
const STEP = 0.45, TRAVEL = 1.1; // seconds between crops, and for each crop's trip

/** Architecture B's two alphas as controls: every pair is refitted on the calibration photo groups and measured on
 *  the test half (scripts/pipeline_scene.py). "Run" sends the scene's five crops through the chosen pair and shows
 *  which bucket each lands in, and which ones moved. */
export function AlphaLab({ rep }: { rep: Report }) {
  const rv = useReveal<HTMLDivElement>(0.8);
  const reduce = !!useReducedMotion();
  const [ood, setOod] = useState(PIPE.served.ood), [cp, setCp] = useState(PIPE.served.cp);
  const [ran, setRan] = useState(cellKey(PIPE.served.ood, PIPE.served.cp));
  const [phase, setPhase] = useState<"rest" | "start" | "go">("rest");
  const timer = useRef<number[]>([]);
  useEffect(() => () => timer.current.forEach(clearTimeout), []);
  const served = cellKey(PIPE.served.ood, PIPE.served.cp), now = cellKey(ood, cp);
  const g = PIPE.grid[now], s = PIPE.grid[served];

  const runIt = () => {
    timer.current.forEach(clearTimeout);
    setRan(now);
    if (reduce) { setPhase("rest"); return; }
    setPhase("start");                                   // back to the start lane, then out to the buckets
    timer.current = [window.setTimeout(() => setPhase("go"), 450),
      window.setTimeout(() => setPhase("rest"), 450 + (STEP * (PIPE.crops.length - 1) + TRAVEL) * 1000)];
  };
  const outcomes = PIPE.crops.map((c) => c.outcomes[ran]);
  const cols = BUCKETS.filter(([o]) => o !== "unclear" || outcomes.includes("unclear"));
  const facts: [string, number, number, (v: number) => string, string?][] = [
    ["Answer holds the true class", g.coverage, s.coverage, (v) => `${(v * 100).toFixed(1)}%`, `target ${pct(1 - cp, 0)}`],
    ["Classes per answer, on average", g.set_size, s.set_size, (v) => v.toFixed(2)],
    ["Wrong with no warning", g.wrong_no_warning, s.wrong_no_warning, (v) => `${(v * 100).toFixed(1)}%`],
    ["Defects shown as nothing", g.silent_miss, s.silent_miss, (v) => `${(v * 100).toFixed(1)}%`],
    ["Undamaged crops shown nothing", g.clean_silent, s.clean_silent, (v) => `${(v * 100).toFixed(1)}%`],
    ["Sent to re-aim", g.reaim, s.reaim, (v) => `${(v * 100).toFixed(1)}%`],
  ];
  const mix: [StateId, string, number][] = ([
    ["detected", "One defect", g.mix.show], ["clear", "No defect", g.mix.nothing], ["ambiguous", "Ambiguous or possible", g.mix.ambiguous + g.mix.possible],
    ["unclear", "Unclear", g.mix.unclear], ["cannot_assess", "Cannot assess", g.mix.reaim],
  ] as [StateId, string, number][]).filter(([, , v]) => v > 0);

  return (
    <div className="panel pl-lab" ref={rv.ref} data-shown={rv.shown}>
      <RuleDraw />
      <h2 className="panel-title rv" style={{ "--i": 0 } as React.CSSProperties}>Try other alphas</h2>
      <p className="panel-sub rv" style={{ "--i": 1 } as React.CSSProperties}>Each choice is refitted the way the team calibrated B, on the {rep.arch_a.cal.families} calibration photo groups, then measured on the {PIPE.test_half.n.toLocaleString("en-US")} test crops. A looser check stops less; a larger conformal α keeps fewer classes, so the visor is quieter and misses more.</p>
      <div className="pl-knobs rv" style={{ "--i": 2 } as React.CSSProperties}>
        <div>
          <p className="pl-knob-label">Unfamiliar-input check α</p>
          <div className="seg" role="radiogroup" aria-label="Unfamiliar-input check alpha">
            {PIPE.ood_alphas.map((v) => <button key={v} type="button" role="radio" aria-checked={v === ood} className="seg-btn" onClick={() => setOod(v)}>{v.toFixed(2)}{v === PIPE.served.ood ? " · served" : ""}</button>)}
          </div>
        </div>
        <div>
          <p className="pl-knob-label">Conformal set α</p>
          <div className="seg" role="radiogroup" aria-label="Conformal set alpha">
            {PIPE.cp_alphas.map((v) => <button key={v} type="button" role="radio" aria-checked={v === cp} className="seg-btn" onClick={() => setCp(v)}>{v.toFixed(2)}{v === PIPE.served.cp ? " · served" : ""}</button>)}
          </div>
        </div>
      </div>
      <dl className="mix-facts pl-facts rv" style={{ "--i": 3 } as React.CSSProperties}>
        {facts.map(([label, v, base, f, note]) => (
          <div key={label}><dt>{label}</dt><dd><AnimatedNumber value={rv.shown ? v : 0} format={(x) => f(x)} />
            <small>{now === served ? (note ?? "served") : `served ${f(base)}`}{note && now !== served ? ` · ${note}` : ""}</small></dd></div>
        ))}
      </dl>
      <div className="mix rv" style={{ "--i": 4 } as React.CSSProperties} role="img" aria-label="Share of test crops given each kind of answer">
        {mix.map(([st, , v]) => (
          <motion.span key={st} className={`mix-seg mix-${st}`} initial={false} animate={{ flexGrow: rv.shown ? v : 0.0001 }} transition={{ duration: rv.t(0.6), ease }}>
            {v >= 0.08 && <span className="fig">{pct(v, 0)}</span>}
          </motion.span>
        ))}
      </div>
      <p className="chart-key rv" style={{ "--i": 4 } as React.CSSProperties}>{mix.map(([st, label, v]) => <span key={st} className="mix-key"><StateGlyph state={st} size={13} /> {label} {pct(v, 0)}</span>)}</p>

      <div className="pl-replay rv" style={{ "--i": 5 } as React.CSSProperties}>
        <div className="pl-replay-head">
          <p><b>The five crops with these alphas.</b> {ran === served ? "Where they land as served." : `Where they land at check α ${ran.split("|")[0]}, conformal α ${ran.split("|")[1]}.`}</p>
          <button type="button" className="btn btn-primary" onClick={runIt} disabled={phase !== "rest"}>Run the 5 crops{now !== served ? " with these alphas" : ""}</button>
        </div>
        <LayoutGroup id="pl-replay">
          <div className="pl-lanes" style={{ "--cols": cols.length } as React.CSSProperties}>
            <div className="pl-start">
              <span className="pl-bucket-head">Start</span>
              {phase === "start" && PIPE.crops.map((c) => <motion.img key={c.outcome} layoutId={`r-${c.outcome}`} src={c.image} alt="" className="pl-rcrop" />)}
              <span className={`pl-sweep${phase === "go" ? " on" : ""}`} style={{ "--dur": `${STEP * (PIPE.crops.length - 1) + TRAVEL}s` } as React.CSSProperties} aria-hidden="true" />
            </div>
            {cols.map(([o, st, label]) => (
              <div key={o} className={`pl-lane tone-${st}`}>
                <span className="pl-bucket-head"><StateGlyph state={st} size={14} /> {label}</span>
                {phase !== "start" && PIPE.crops.map((c, i) => outcomes[i] === o && (
                  <motion.figure key={c.outcome} layoutId={`r-${c.outcome}`} className="pl-rfig"
                    transition={{ duration: reduce ? 0 : TRAVEL, delay: phase === "go" ? i * STEP : 0, ease }}>
                    <img src={c.image} alt={`Crop ${i + 1}: lands in ${bucketLabel(o)}`} className="pl-rcrop" />
                    {c.outcomes[served] !== o && <figcaption>was {bucketLabel(c.outcomes[served]).toLowerCase()}</figcaption>}
                  </motion.figure>
                ))}
              </div>
            ))}
          </div>
        </LayoutGroup>
        <p className="coverage-ref">{PIPE.crops.filter((c) => c.outcomes[ran] !== c.outcomes[served]).length === 0 ? "Every crop lands where it does as served." : `${PIPE.crops.filter((c) => c.outcomes[ran] !== c.outcomes[served]).length} of 5 crops land in a different bucket than as served.`}{PIPE.crops.some((c) => c.truth !== "no-defect" && !isOff(c) && c.outcomes[ran] === "nothing") ? " A real defect is now shown as nothing." : ""}{PIPE.crops.some((c) => isOff(c) && c.outcomes[ran] !== "reaim") ? " The glove now gets through the check and is shown to the inspector." : ""}</p>
        {PIPE.crops.filter((c) => c.credit).map((c) => <p key={c.outcome} className="pl-credit-line">Glove photo: <a href={c.credit!.url} target="_blank" rel="noreferrer">{c.credit!.author}</a>, <a href={c.credit!.license_url} target="_blank" rel="noreferrer">{c.credit!.license}</a>, from {c.credit!.source}.</p>)}
      </div>
    </div>
  );
}
