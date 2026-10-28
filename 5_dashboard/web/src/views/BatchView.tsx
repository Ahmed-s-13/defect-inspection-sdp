import type React from "react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useLayoutEffect, useRef, useState, type DragEvent, type ReactNode } from "react";
import { getSamples, sampleFile } from "../api/client";
import { AnimatedNumber } from "../components/AnimatedNumber";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { Tile } from "../components/Tile";
import { filesFromDrop, isImage } from "../lib/drop";
import { className, int, pct } from "../lib/format";
import { parseLabelsCsv } from "../lib/labels";
import { accuracy, answerMisses, answerStats, confusion, macroF1, perClass, wilson } from "../lib/metrics";
import { useBench, type BatchFilter } from "../state/bench";
import "../styles/batch.css";

const ease = [0.22, 1, 0.36, 1] as const;

/** Swaps between two states of a figure or line ("—" and a number) with a short crossfade. */
function Swap({ k, children }: { k: string; children: ReactNode }) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.span key={k} className="swap" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.22, ease }}>
        {children}
      </motion.span>
    </AnimatePresence>
  );
}

let lastScroll = 0; // survives the view unmounting while Inspect is open

const FILTERS: { id: BatchFilter; label: string }[] = [
  { id: "all", label: "All" }, { id: "errors", label: "Wrong" }, { id: "detected", label: "Defect detected" }, { id: "clear", label: "No defect" },
  { id: "ambiguous", label: "Ambiguous" }, { id: "unclear", label: "Unclear" }, { id: "retake", label: "Bad image" },
  { id: "cannot_assess", label: "Cannot assess" },
];

export function BatchView({ classes, onOpen }: { classes: string[]; onOpen(id: string): void }) {
  const bench = useBench();
  const [nSamples, setNSamples] = useState<number | null>(null);
  useEffect(() => { getSamples().then((s) => setNSamples(s.length)).catch(() => setNSamples(null)); }, [classes.length]);
  const samplesLabel = nSamples ? `the ${nSamples} test samples` : "the test samples";
  const items = bench.state.batch.map((id) => bench.state.items[id]);
  const { filter, row } = bench.batchUi;
  const setFilter = (f: BatchFilter) => bench.setBatchUi((u) => ({ ...u, filter: f }));
  const setRow = (r: number | null) => bench.setBatchUi((u) => ({ ...u, row: r }));
  const [over, setOver] = useState(false);
  // keep your place: remember where the list was scrolled and return there when coming back from Inspect
  useLayoutEffect(() => {
    window.scrollTo(0, lastScroll);
    const onScroll = () => { lastScroll = window.scrollY; };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  const filesIn = useRef<HTMLInputElement>(null);
  const dirIn = useRef<HTMLInputElement>(null);

  const add = async (images: File[], labelsFile?: File) => {
    // raw labels: they are matched to the model's classes in the store, even if the model is still loading
    const csv = labelsFile ? parseLabelsCsv(await labelsFile.text(), []) : new Map<string, string>();
    bench.addBatch(images.map((file) => ({ file, csvLabel: csv.get(file.name) })));
  };
  const fromInput = (list: FileList | null) => {
    const all = Array.from(list ?? []);
    void add(all.filter(isImage), all.find((f) => f.name.toLowerCase() === "labels.csv"));
  };
  const drop = async (e: DragEvent) => { e.preventDefault(); setOver(false); const { images, labels } = await filesFromDrop(e.dataTransfer); void add(images, labels); };
  const runSamples = async () => {
    const s = await getSamples();
    const files = await Promise.all(s.map((x) => sampleFile(x.file)));
    bench.addBatch(files.map((file, i) => ({ file, csvLabel: s[i].cls })));
  };

  const done = items.filter((i) => i.result);
  const pairs = done.filter((i) => i.truth).map((i) => ({ truth: i.truth!.cls, pred: i.result!.top }));
  const right = pairs.filter((p) => p.truth === p.pred).length;
  const [lo, hi] = wilson(right, pairs.length);
  const cm = confusion(pairs, classes);       // recomputed every render: a relabel must show at once
  const recall = perClass(pairs, classes);
  const ans = answerStats(done.filter((i) => i.truth).map((i) => ({ r: i.result!, truth: i.truth!.cls })));
  const count = (s: string) => done.filter((i) => i.result!.state === s).length;
  const meanMs = done.length ? done.reduce((a, i) => a + i.result!.ms, 0) / done.length : NaN;

  const shown = items.filter((i) => {
    if (row != null && i.truth?.cls !== classes[row]) return false;
    if (filter === "all") return true;
    if (filter === "errors") return !!(i.result && i.truth && answerMisses(i.result, i.truth.cls));
    return i.result?.state === filter;
  });

  const exportCsv = () => {
    const head = "file,state,answer,reason,top_class,calibrated_confidence,p_quality,p_ood,true,top_correct,answer_holds_true";
    const lines = items.map((i) => {
      const r = i.result;
      return [JSON.stringify(i.name), r?.state ?? i.status, r ? r.answer.join(";") : "", r?.reason ? JSON.stringify(r.reason) : "",
        r?.top ?? "", r ? r.confidence.toFixed(4) : "", r?.p_quality?.toFixed(4) ?? "", r?.p_ood?.toFixed(4) ?? "",
        i.truth?.cls ?? "", r && i.truth ? String(i.truth.cls === r.top) : "",
        r && i.truth && r.answer.length ? String(r.answer.includes(i.truth.cls)) : ""].join(",");
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([[head, ...lines].join("\n")], { type: "text/csv" }));
    a.download = `batch-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <section className={`batch${over ? " is-over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={drop}>
      <header className="batch-head enter" style={{ "--i": 0 } as React.CSSProperties}>
        <div>
          <h1 className="view-title">Batch</h1>
          <p className="view-sub">{items.length
            ? `${int(done.length)} of ${int(items.length)} images classified · ${int(pairs.length)} labelled`
            : `Drop a folder of images, or run ${samplesLabel}${classes.includes("no-defect") ? " (5 per class, no defect included)" : ""}. File names that start with a class name, or a labels.csv, count as labels.`}</p>
        </div>
        <div className="batch-actions">
          <button className="btn" onClick={() => filesIn.current?.click()}>Add images</button>
          <button className="btn" onClick={() => dirIn.current?.click()}>Add folder</button>
          <button className="btn btn-primary" onClick={runSamples}>Run {nSamples ? `the ${nSamples} samples` : "the samples"}</button>
          <button className="btn" onClick={exportCsv} disabled={!done.length}>Export CSV</button>
          <button className="btn btn-quiet" onClick={() => { bench.clearBatch(); setRow(null); }} disabled={!items.length}>Clear</button>
        </div>
        <input ref={filesIn} type="file" multiple hidden accept="image/*,.csv" onChange={(e) => { fromInput(e.target.files); e.target.value = ""; }} />
        <input ref={dirIn} type="file" hidden {...({ webkitdirectory: "" } as object)} onChange={(e) => { fromInput(e.target.files); e.target.value = ""; }} />
      </header>

      <div className="summary enter" style={{ "--i": 1 } as React.CSSProperties} role="group" aria-label="Batch results">
        <div className="summary-cell">
          <span className="summary-label">Top class right</span>
          <span className="summary-fig"><Swap k={pairs.length ? "n" : "none"}>{pairs.length ? <AnimatedNumber value={accuracy(pairs) * 100} format={(v) => `${v.toFixed(1)}%`} /> : <span className="fig">—</span>}</Swap></span>
          <span className="summary-sub"><Swap k={pairs.length ? "n" : "none"}>{pairs.length ? `${right} of ${pairs.length} · 95% interval ${pct(lo, 0)}–${pct(hi, 0)} · macro-F1 ${macroF1(pairs, classes).toFixed(2)}` : "No labelled images yet"}</Swap></span>
          {pairs.length > 0 && pairs.length < 30 && <span className="summary-note">Too few images to compare models.</span>}
        </div>
        <div className="summary-cell">
          <span className="summary-label">Answer holds the true class</span>
          <span className="summary-fig"><Swap k={ans.answered ? "n" : "none"}>{ans.answered ? <AnimatedNumber value={(ans.holds / ans.answered) * 100} format={(v) => `${v.toFixed(1)}%`} /> : <span className="fig">—</span>}</Swap></span>
          <span className="summary-sub"><Swap k={ans.answered ? "n" : "none"}>{ans.answered ? `${ans.holds} of ${ans.answered} answered · ${ans.silent} wrong with no warning` : "Target about 90% on normal photos"}</Swap></span>
        </div>
        <div className="summary-cell">
          <span className="summary-label">Needs a person</span>
          <span className="summary-fig"><AnimatedNumber value={done.filter((i) => i.result!.state !== "detected" && i.result!.state !== "clear").length} format={(v) => int(Math.round(v))} /></span>
          <span className="summary-sub">{count("ambiguous") + count("unclear")} ambiguous{count("retake") > 0 ? ` · ${count("retake")} bad image` : ""} · {count("cannot_assess")} cannot assess</span>
        </div>
        <div className="summary-cell">
          <span className="summary-label">Time per image</span>
          <span className="summary-fig"><Swap k={Number.isFinite(meanMs) ? "n" : "none"}>{Number.isFinite(meanMs) ? <AnimatedNumber value={meanMs} format={(v) => `${Math.round(v)} ms`} /> : <span className="fig">—</span>}</Swap></span>
          <span className="summary-sub">8 views each, on the server</span>
        </div>
      </div>

      <div className="batch-body">
        <div className="enter" style={{ "--i": 2 } as React.CSSProperties}>
          <div className="seg" role="radiogroup" aria-label="Show">
            {FILTERS.filter((f) => (f.id !== "unclear" || count("unclear") > 0) && (f.id !== "retake" || count("retake") > 0) && (f.id !== "clear" || classes.includes("no-defect"))).map((f) => (
              <button key={f.id} role="radio" aria-checked={filter === f.id} className="seg-btn" onClick={() => setFilter(f.id)}>{f.label}</button>
            ))}
            {row != null && <button className="seg-clear" onClick={() => setRow(null)}>True class: {className(classes[row])} ×</button>}
          </div>
          <AnimatePresence mode="wait" initial={false}>
            {items.length === 0 ? (
              <motion.div key="empty" className="batch-empty" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, transition: { duration: 0.18 } }} transition={{ duration: 0.35, ease }}>
                <p>Nothing here yet. Drop images anywhere on this page.</p>
              </motion.div>
            ) : shown.length === 0 ? (
              <motion.div key="nomatch" className="batch-empty" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, transition: { duration: 0.18 } }} transition={{ duration: 0.3, ease }}>
                <p>No images match this filter.</p>
              </motion.div>
            ) : (
              <motion.div key="grid" className="grid" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.985, transition: { duration: 0.28, ease } }} transition={{ duration: 0.2 }}>
                <AnimatePresence mode="popLayout" initial={false}>
                  {shown.map((it, i) => <Tile key={it.id} item={it} index={i} onOpen={() => onOpen(it.id)} />)}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <aside className="batch-side enter" style={{ "--i": 3 } as React.CSSProperties}>
          <h2 className="panel-title">Where it goes wrong</h2>
          <p className="side-sub">The model's top class, even when the answer was a set or a check stopped the image.</p>
          <AnimatePresence mode="wait" initial={false}>
          {pairs.length ? (
            <motion.div key="matrix" className="side-stack" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, transition: { duration: 0.2 } }} transition={{ duration: 0.35, ease }}>
              <ConfusionMatrix classes={classes} matrix={cm} selected={row} onSelect={setRow} label="Batch confusion matrix, row percentages" />
              <ul className="recall-list">
                {recall.filter((r) => r.support).map((r) => (
                  <li key={r.cls}><span>{className(r.cls)}</span><span className="fig">{pct(r.recall, 0)}</span><span className="recall-n">of {r.support}</span></li>
                ))}
              </ul>
            </motion.div>
          ) : (
            <motion.p key="hint" className="muted" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, transition: { duration: 0.2 } }} transition={{ duration: 0.35, ease }}>
              Label images (file names, labels.csv, or the True class menu in Inspect) to see this.
            </motion.p>
          )}
          </AnimatePresence>
        </aside>
      </div>
    </section>
  );
}
