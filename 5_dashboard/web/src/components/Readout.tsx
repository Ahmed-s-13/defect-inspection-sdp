import type React from "react";
import { AnimatePresence, motion } from "motion/react";
import type { Prediction } from "../api/client";
import { className, int, pct } from "../lib/format";
import type { Item } from "../state/bench";
import { SoftmaxBars } from "./SoftmaxBars";
import { STATE_WORDS, StateGlyph, actionLine, answerText } from "./StateBadge";
import { ViewsStrip } from "./ViewsFan";

/** What the answer can't tell you: the 6-class legacy model has no "no defect" class; gc7 has one. */
const caveat = (classes: string[]) => (classes.includes("no-defect")
  ? "No defect speaks for this square only. When the answer is \u201cPossible\u201d, a defect and no defect are both kept: a person should look."
  : 'This model assumes damage is present and names its type. It has no "no defect" class.');
const glide = { type: "spring" as const, stiffness: 380, damping: 36 };

/** One sentence on how far to trust this answer: the decision layer's guarantee, or which check stopped it. */
function trustLine(r: Prediction) {
  const cover = r.alpha != null ? Math.round((1 - r.alpha) * 100) : 90;
  if (r.state === "retake") return <>The image-quality check stopped this crop before any class was shown. Only <span className="fig">{pct(r.p_quality ?? 0, 0)}</span> of normal photos look this unusual.</>;
  if (r.state === "cannot_assess") return <>The unfamiliar-input check stopped this crop before any class was shown. Only <span className="fig">{pct(r.p_ood ?? 0, 0)}</span> of normal photos are this far from the training photos.</>;
  if (r.state === "unrated") return <>Confidence <span className="fig">{pct(r.confidence)}</span>. This model has no calibrated decision layer, so the raw number is shown.</>;
  return <>Percentages are calibrated. On normal photos the answer holds the true class about <span className="fig">{cover}%</span> of the time{r.state === "detected" || r.state === "clear" ? ", and here only one class is likely enough to keep" : ""}.</>;
}

/** A gate's result in words: passed or stopped, with its p-value. */
function gate(p: number | null, alpha: number | null) {
  if (p == null || alpha == null) return <span className="muted">not run</span>;
  return <>{p <= alpha ? <b>Stopped</b> : "Passed"} <small>p = {p.toFixed(2)}, stops at {alpha.toFixed(2)} or below</small></>;
}

export function Readout({ item, fanning, classes, onTruth, session }: {
  item: Item | null; fanning: boolean; classes: string[]; onTruth(cls: string | null): void; session: { right: number; n: number };
}) {
  const r = item?.result;
  const running = item?.status === "running";
  const fault = item?.status === "error" && item.errorState === "system_fault";

  return (
    <aside className="readout enter" style={{ "--i": 1 } as React.CSSProperties} aria-live="polite" aria-busy={running}>
      <AnimatePresence mode="wait" initial={false}>
        {!item ? (
          <motion.div key="legend" className="readout-stack" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6, transition: { duration: 0.2 } }} transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}>
        <h2 className="readout-title">What the answers mean</h2>
        <ul className="legend">
          {!classes.includes("no-defect") && <li><StateGlyph state="retake" size={18} /><span><b>Bad image</b>: blurry, too dark, too bright or glare. Re-take it.</span></li>}
          <li><StateGlyph state="cannot_assess" size={18} /><span><b>Cannot assess</b>: unlike any training photo. Re-aim.</span></li>
          <li><StateGlyph state="detected" size={18} /><span><b>Defect detected</b>: only one defect class is likely enough to keep.</span></li>
          {classes.includes("no-defect") && <li><StateGlyph state="clear" size={18} /><span><b>No defect</b>: only &ldquo;no defect&rdquo; is likely enough to keep. The visor shows nothing.</span></li>}
          <li><StateGlyph state="ambiguous" size={18} /><span><b>Ambiguous</b>: two or more classes are kept{classes.includes("no-defect") ? <>; <b>Possible</b> when one of them is &ldquo;no defect&rdquo;</> : null}. A person should check.</span></li>
        </ul>
        <p className="legend-note">On normal photos the answer holds the true class about 90% of the time. {classes.includes("no-defect") ? "The unfamiliar-input check also stops about 3 in 100 normal photos; there is no image-quality check." : "To catch most bad images, each check also stops about 1 in 10 normal photos."}</p>
        <p className="caveat">{caveat(classes)}</p>
          </motion.div>
        ) : (
          <motion.div key="result" className="readout-stack" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6, transition: { duration: 0.2 } }} transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}>
      {fault || item.status === "error" ? (
        <div className="verdict-block">
          <h2 className={`verdict verdict-${fault ? "system_fault" : "cannot_assess"}`}>
            <span className="verdict-state"><StateGlyph state={fault ? "system_fault" : "cannot_assess"} size={26} /> {fault ? STATE_WORDS.system_fault : STATE_WORDS.cannot_assess}</span>
          </h2>
          <p className="fault-text">{item.error}</p>
        </div>
      ) : r ? (
        <div className="verdict-block">
          <h2 className={`verdict verdict-${r.state}`}>
            <span className="verdict-state"><StateGlyph state={r.state} size={26} /> {STATE_WORDS[r.state]}</span>
            {answerText(r) && <span className={`verdict-class${r.answer.length > 1 ? " is-set" : ""}`}>{answerText(r)}{(r.state === "detected" || r.state === "clear") && <span className="fig verdict-pct"> {pct(r.confidence)}</span>}</span>}
          </h2>
          {actionLine(r) && <p className="verdict-sub">{actionLine(r)}</p>}
          <AnimatePresence mode="wait" initial={false}>
            <motion.p key={`${r.state}-${r.top}-${r.confidence}`} className="reliability" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
              {trustLine(r)}
            </motion.p>
          </AnimatePresence>
        </div>
      ) : (
        <div className="verdict-block"><p className="verdict-wait">Reading the image…</p></div>
      )}

      {r && <SoftmaxBars ranked={r.ranked} state={r.state} answer={r.answer} threshold={r.set_threshold} />}
      <ViewsStrip itemId={item.id} run={item.run} views={item.detail?.views ?? null} fanning={fanning} />

      <motion.dl layout="position" className="facts" transition={glide}>
        {r && r.state !== "unrated" && r.alpha_quality != null && <div><dt>Image-quality check</dt><dd>{gate(r.p_quality, r.alpha_quality)}</dd></div>}
        {r && r.state !== "unrated" && <div><dt>Unfamiliar-input check</dt><dd>{gate(r.p_ood, r.alpha_ood)}</dd></div>}
        {r && <div><dt>Time</dt><dd className="fig">{Math.round(r.ms)} ms</dd></div>}
        <div>
          <dt><label htmlFor="truth">True class</label></dt>
          <dd>
            <select id="truth" className="truth" value={item.truth?.cls ?? ""} onChange={(e) => onTruth(e.target.value || null)}>
              <option value="">Not labelled</option>
              {classes.map((c) => <option key={c} value={c}>{className(c)}</option>)}
            </select>
            {item.truth && item.truth.source !== "manual" && <small> from {item.truth.source === "csv" ? "labels.csv" : "the file name"}</small>}
          </dd>
        </div>
        {session.n > 0 && <div><dt>This session</dt><dd><span className="fig">{session.right} of {session.n}</span> labelled images right</dd></div>}
      </motion.dl>
      <motion.p layout="position" className="caveat" transition={glide}>{caveat(classes)}</motion.p>
          </motion.div>
        )}
      </AnimatePresence>
    </aside>
  );
}
