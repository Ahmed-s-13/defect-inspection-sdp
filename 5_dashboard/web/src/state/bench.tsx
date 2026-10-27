import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from "react";
import { batchStream, predict, predictionFromLine, type PredictDetail, type Prediction, type Square } from "../api/client";
import { truthFor } from "../lib/labels";
import { latestGate } from "../lib/requests";

export interface Truth { cls: string; source: "csv" | "filename" | "manual" }
export interface Item {
  id: string; name: string; file: File; url: string; truth?: Truth;
  csvLabel?: string; // raw labels.csv class, resolved against the model's classes once they are known
  status: "queued" | "running" | "done" | "error";
  result?: Prediction; detail?: PredictDetail; thumb?: string; error?: string; errorState?: string;
  run: number; // increments per Inspect prediction, so each run's 8-view fan has its own layout ids
}
interface State { items: Record<string, Item>; batch: string[]; inspectId: string | null; inspectFrom: "batch" | null }
type Action =
  | { type: "add"; items: Item[]; batch: boolean; inspect?: string }
  | { type: "patch"; id: string; patch: Partial<Item> }
  | { type: "inspect"; id: string | null; from?: "batch" | null }
  | { type: "clearBatch" };

let seq = 0;
export function makeItem(file: File, csvLabel?: string, classes: string[] = []): Item {
  return { id: `i${++seq}`, name: file.name, file, url: URL.createObjectURL(file), csvLabel,
    truth: truthFor(file.name, csvLabel, classes), status: "queued", run: 0 };
}

export type BatchFilter = "all" | "errors" | "detected" | "clear" | "ambiguous" | "unclear" | "retake" | "cannot_assess";
export interface BatchUi { filter: BatchFilter; row: number | null }

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case "add": {
      const items = { ...s.items };
      for (const it of a.items) items[it.id] = it;
      return { items, batch: a.batch ? [...s.batch, ...a.items.map((i) => i.id)] : s.batch, inspectId: a.inspect ?? s.inspectId,
        inspectFrom: a.inspect ? null : s.inspectFrom };
    }
    case "patch": {
      const cur = s.items[a.id];
      return cur ? { ...s, items: { ...s.items, [a.id]: { ...cur, ...a.patch } } } : s;
    }
    case "inspect":
      return { ...s, inspectId: a.id, inspectFrom: a.id ? a.from ?? null : null };
    case "clearBatch": {
      const items = { ...s.items };
      for (const id of s.batch) if (id !== s.inspectId) { URL.revokeObjectURL(items[id].url); delete items[id]; }
      return { ...s, items, batch: [], inspectFrom: null };
    }
  }
}

function useBenchState() {
  const [state, dispatch] = useReducer(reducer, { items: {}, batch: [], inspectId: null, inspectFrom: null });
  const ref = useRef(state);
  useEffect(() => { ref.current = state; }, [state]);
  const patch = useCallback((id: string, p: Partial<Item>) => dispatch({ type: "patch", id, patch: p }), []);
  const gate = useRef(latestGate());
  const classes = useRef<string[]>([]);
  const [batchUi, setBatchUi] = useState<BatchUi>({ filter: "all", row: null });

  /** Labels dropped while the model was still loading are resolved as soon as its classes are known. */
  const setClasses = useCallback((cls: string[]) => {
    classes.current = cls;
    for (const it of Object.values(ref.current.items)) {
      if (it.truth) continue;
      const t = truthFor(it.name, it.csvLabel, cls);
      if (t) patch(it.id, { truth: t });
    }
  }, [patch]);

  const predictInspect = useCallback(async (id: string, square?: Square) => {
    const it = ref.current.items[id];
    if (!it) return;
    const token = gate.current.next(id); // only the newest reply for this image may land
    patch(id, { status: "running", error: undefined, errorState: undefined, run: it.run + 1 });
    ref.current = { ...ref.current, items: { ...ref.current.items, [id]: { ...it, run: it.run + 1 } } };
    try {
      const { prediction, detail } = await predict(it.file, square);
      if (gate.current.isCurrent(id, token)) patch(id, { status: "done", result: prediction, detail });
    } catch (e) {
      const err = e as Error & { state?: string };
      if (gate.current.isCurrent(id, token)) patch(id, { status: "error", error: err.message, errorState: err.state });
    }
  }, [patch]);

  const inspectFile = useCallback((file: File, csvLabel?: string) => {
    const it = makeItem(file, csvLabel, classes.current);
    dispatch({ type: "add", items: [it], batch: false, inspect: it.id });
    ref.current = { ...ref.current, items: { ...ref.current.items, [it.id]: it } };
    void predictInspect(it.id);
  }, [predictInspect]);

  const runBatch = useCallback(async (ids: string[]) => {
    const files = ids.map((id) => ref.current.items[id].file);
    for (const id of ids) patch(id, { status: "running" });
    const answered = new Set<number>();
    const unanswered = (error: string, errorState?: string) =>
      ids.forEach((id, i) => { if (!answered.has(i)) patch(id, { status: "error", error, errorState }); });
    try {
      await batchStream(files, (line) => {
        answered.add(line.index);
        const id = ids[line.index];
        if (line.ok) patch(id, { status: "done", result: predictionFromLine(line), thumb: line.thumb });
        else patch(id, { status: "error", error: line.error });
      });
      unanswered("No answer from the server for this image"); // a stream that ended early never leaves tiles waiting
    } catch (e) {
      const err = e as Error & { state?: string };
      unanswered(err.message, err.state);
    }
  }, [patch]);

  const addBatch = useCallback((entries: { file: File; csvLabel?: string }[]) => {
    const items = entries.map((e) => makeItem(e.file, e.csvLabel, classes.current));
    dispatch({ type: "add", items, batch: true });
    ref.current = { ...ref.current, items: { ...ref.current.items, ...Object.fromEntries(items.map((i) => [i.id, i])) } };
    void runBatch(items.map((i) => i.id));
  }, [runBatch]);

  const openInInspect = useCallback((id: string) => {
    dispatch({ type: "inspect", id, from: "batch" }); // Inspect then offers a way back to the batch
    const it = ref.current.items[id];
    if (it && !it.detail) void predictInspect(id);
  }, [predictInspect]);

  const setTruth = useCallback((id: string, cls: string | null) =>
    patch(id, { truth: cls ? { cls, source: "manual" } : undefined }), [patch]);
  const clearBatch = useCallback(() => dispatch({ type: "clearBatch" }), []);
  const closeInspect = useCallback(() => dispatch({ type: "inspect", id: null }), []);

  return useMemo(() => ({ state, inspectFile, predictInspect, addBatch, runBatch, openInInspect, setTruth, clearBatch, closeInspect, setClasses, batchUi, setBatchUi }),
    [state, inspectFile, predictInspect, addBatch, runBatch, openInInspect, setTruth, clearBatch, closeInspect, setClasses, batchUi]);
}

type Bench = ReturnType<typeof useBenchState>;
const Ctx = createContext<Bench | null>(null);
export function BenchProvider({ children }: { children: ReactNode }) {
  return <Ctx.Provider value={useBenchState()}>{children}</Ctx.Provider>;
}
export function useBench() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useBench outside BenchProvider");
  return v;
}
