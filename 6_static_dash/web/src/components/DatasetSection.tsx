import { motion } from "motion/react";
import { useState } from "react";
import type { StateId } from "../api/client";
import ds from "../data/datasets.json";
import { className, pct } from "../lib/format";
import { useReveal } from "../lib/useReveal";
import { RuleDraw } from "./RuleDraw";
import { STATE_WORDS, StateGlyph } from "./StateBadge";

/** scripts/uncertainty_data.py: every dataset we used, its size, what it was used for, and samples with
 *  architecture B's answer (as Inspect would give it). */
interface DsSample { image: string; label: string; state: StateId; answer: string[]; top: string; confidence: number; credit?: { author: string; url: string; license: string } }
interface Ds { name: string; id: string; size: string; used: string; detail?: string; licence?: string; categories?: number; fit?: number; test?: number;
  outcome?: Record<string, number>; outcome_n?: number; samples: DsSample[] }
const DATA = ds as unknown as { aircraft: Ds[]; nonaircraft: Ds[] };
const ease = [0.22, 1, 0.36, 1] as const;

const answerWords = (s: DsSample) => (s.state === "clear" ? "Nothing shown" : s.state === "cannot_assess" ? "Re-aim"
  : s.state === "ambiguous" && s.answer.includes("no-defect") ? `Possible ${s.answer.filter((c) => c !== "no-defect").map((c) => className(c).toLowerCase()).join(" or ")}`
  : s.answer.length > 1 ? s.answer.map(className).join(" or ") : `${className(s.answer[0] ?? s.top)} ${pct(s.confidence, 0)}`);
const label = (l: string) => (l.startsWith("texture: ") ? l.slice(9) : l.replace(/^[a-z_]+\//, "").replace(/_/g, " "));
const outcomeLine = (o?: Record<string, number>, n?: number) => {
  if (!o || !Object.keys(o).length) return null;
  const reaim = o["re-aim"] ?? 0, shown = (o["show class"] ?? 0) + (o.ambiguous ?? 0) + (o["possible defect"] ?? 0);
  return <>On {n?.toLocaleString("en-US")} crops, B re-aims <b className="fig">{pct(reaim, 1)}</b>; {pct(shown, 1)} reach the visor as a defect or possible defect.</>;
};

function Row({ d, i, open, onToggle, shown, nonAircraft }: { d: Ds; i: number; open: boolean; onToggle(): void; shown: boolean; nonAircraft: boolean }) {
  return (
    <>
      <motion.tr className={`ds-row${open ? " is-open" : ""}`} initial={false} animate={{ opacity: shown ? 1 : 0, y: shown ? 0 : 10 }}
        transition={{ duration: 0.45, delay: shown ? 0.06 * i : 0, ease }}>
        <th scope="row"><button type="button" className="ds-toggle" aria-expanded={open} onClick={onToggle}><span className="ds-caret" aria-hidden="true">{open ? "−" : "+"}</span>{d.name}</button></th>
        <td className="fig">{d.size}</td>
        <td>{d.used}</td>
        {nonAircraft && <td className="fig">{d.fit !== undefined && d.test !== undefined
          ? <>{d.fit.toLocaleString("en-US")} train / {d.test.toLocaleString("en-US")} test</>
          : <>0 train / {d.outcome_n?.toLocaleString("en-US") ?? "all"} test</>}</td>}
        {nonAircraft && <td className="ds-out">{d.outcome && Object.keys(d.outcome).length ? <><span className="fig">{pct(d.outcome["re-aim"] ?? 0, 1)}</span> re-aimed</> : "see below"}</td>}
      </motion.tr>
      {open && (
        <tr className="ds-detail">
          <td colSpan={nonAircraft ? 5 : 3}>
            <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease }}>
              {d.detail && <p className="ds-note">{d.detail}.</p>}
              {nonAircraft && <p className="ds-note">{d.categories ? <>{d.categories} categories: {d.fit?.toLocaleString("en-US")} crops in training categories, {d.test?.toLocaleString("en-US")} in test categories (left out when we tested the check on kinds of photo it had not seen). </> : null}Licence: {d.licence}. {outcomeLine(d.outcome, d.outcome_n)}</p>}
              <ul className="ds-samples">
                {d.samples.map((s, j) => (
                  <motion.li key={s.image} initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.3, delay: 0.05 * j, ease }}>
                    <img src={s.image} alt={label(s.label)} loading="lazy" />
                    <span className="ds-label">{label(s.label)}</span>
                    <span className={`ds-answer tone-${s.state}`}><StateGlyph state={s.state} size={13} /> {answerWords(s)}</span>
                  </motion.li>
                ))}
              </ul>
              <p className="ds-key">Each sample's answer is architecture B's, as the Inspect page would give it ({STATE_WORDS.cannot_assess.toLowerCase()} = re-aim).</p>
            </motion.div>
          </td>
        </tr>
      )}
    </>
  );
}

function Table({ rows, nonAircraft, caption }: { rows: Ds[]; nonAircraft: boolean; caption: string }) {
  const rv = useReveal<HTMLDivElement>(0.85);
  const [open, setOpen] = useState<string | null>(null); // static dash: every row starts closed
  return (
    <div className="ds-wrap" ref={rv.ref} data-shown={rv.shown}>
      <table className="ds-table">
        <caption className="sr-only">{caption}</caption>
        <thead><tr><th scope="col">Dataset</th><th scope="col">Size</th><th scope="col">Where we used it</th>{nonAircraft && <><th scope="col">Train / test crops</th><th scope="col">B's answer</th></>}</tr></thead>
        <tbody>
          {rows.map((d, i) => <Row key={d.id} d={d} i={i} open={open === d.id} onToggle={() => setOpen(open === d.id ? null : d.id)} shown={rv.shown} nonAircraft={nonAircraft} />)}
        </tbody>
      </table>
    </div>
  );
}

/** The datasets, in numbers: the aircraft ones, then the non-aircraft ones behind the aircraft-or-not check. */
export function DatasetSection({ first, dimAt }: { first: number; dimAt(...steps: number[]): boolean }) {
  return (
    <>
      <section className="ds-sec present-dim" data-step={first} data-dim={dimAt(first)}>
        <RuleDraw />
        <h3 className="data-h2">Aircraft photos</h3>
        <p className="how-sub">Open a row to see samples and what architecture B answers for each.</p>
        <Table rows={DATA.aircraft} nonAircraft={false} caption="Aircraft datasets" />
      </section>
      <section className="ds-sec present-dim" data-step={first + 1} data-dim={dimAt(first + 1)}>
        <RuleDraw />
        <p className="how-sub">For Open Images and Places365, we take the largest square centred in each photo as a centre crop. Open Images also provides labelled object boxes, which let us crop around objects such as helmets, hands and tools with surrounding context. Crops are resized to 224 × 224 pixels.</p>
        <h3 className="data-h2">Not aircraft: what the aircraft-or-not check learned from</h3>
        <p className="how-sub">The train / test column shows the category splits used in our experiments. The controlled AoN comparison trains on 2,566 Open Images and Places365 crops and tests on 2,170 crops from different categories, plus two crops of your helmet photo (2,172 test crops total). The deployed AoN learns from both Open Images and Places365 splits, 4,736 crops in total, so those test categories are not held out from the deployed model. DTD and plain surfaces were tried separately and excluded from deployment.</p>
        <Table rows={DATA.nonaircraft} nonAircraft caption="Non-aircraft datasets" />
      </section>
    </>
  );
}
export const DATASET_STEPS = 2;
