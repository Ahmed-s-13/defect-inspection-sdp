import { chunkWithOffsets } from "../lib/requests";
/** Architecture B's answers: one defect class, "no defect" alone (clear, nothing on the visor), a set of 2+, an empty set, or a gate that
 *  stopped the crop. */
export type StateId = "detected" | "clear" | "ambiguous" | "unclear" | "retake" | "cannot_assess" | "unrated";
export type UiState = StateId | "system_fault";
export interface Square { left: number; top: number; side: number }
export interface Ranked { cls: string; p: number }
/** probs/ranked are architecture B's calibrated (tempered) percentages; `answer` is its conformal set, empty
 *  when a gate stopped the crop. p_quality / p_ood are the gates' p-values (null when the gate did not run). */
export interface Prediction {
  classes: string[]; probs: number[]; ranked: Ranked[]; top: string; confidence: number; state: StateId;
  answer: string[]; reason: string | null; p_quality: number | null; p_ood: number | null;
  set_threshold: number | null; alpha: number | null; alpha_quality: number | null; alpha_ood: number | null; ms: number;
}
export interface PredictDetail { image: { width: number; height: number }; square: Square; crop: string; views: string[] }
export interface BatchLine extends Partial<Prediction> { index: number; name: string; ok: boolean; error?: string; thumb?: string }
export interface ModelSummary {
  id: string; label: string; description: string; trained_on: string; leak_free: boolean;
  available: boolean; reason: string | null; is_main: boolean;
}
export interface ModelsInfo { main: string; active: string; device: string; loaded: boolean; classes: string[] | null; models: ModelSummary[] }
export interface Sample { file: string; cls: string }
export interface Bin { lo: number; hi: number; n: number; confidence: number; accuracy: number }
export interface Report {
  model: string; label: string; description: string; trained_on: string; computed_at: string;
  classes: string[]; target_macro_f1: number;
  test: {
    n: number; groups: number; accuracy: number; macro_f1: number; balanced_accuracy: number;
    ci: [number, number] | null; confusion: number[][];
    per_class: { cls: string; precision: number; recall: number; f1: number; support: number }[];
  };
  val: { n: number; accuracy: number };
  /** the model it replaced: for gc7, the 6-class v2 on the same defect crops (defects_only) */
  baseline: { accuracy: number; macro_f1: number; recall: Record<string, number>; label: string; defects_only: boolean } | null;
  no_defect: NoDefect | null;
  backbone_epoch: number | null; head: { knn_weight: number; tau: number } | null; val_macro_f1: number | null;
  d5: { accuracy: number; base_accuracy: number; n: number; counts: Record<string, number> } | null;
  arch_a: ArchA;
  provenance: { weights: { file: string; bytes: number; modified: string }[]; features_cache: string; summary: string; split: string };
}

/** What the 7th class adds, on the test split (gc7 only). Source keys: healed, clean_cropper. */
export interface NoDefect {
  n: number; n_defects: number; recall: number; recall_by_source: Record<string, number>; n_by_source: Record<string, number>;
  defects_called_clear: number; defects_called_clear_by_class: Record<string, number>;
  defects_best: { accuracy: number; macro_f1: number }; v2_defects: { accuracy: number; macro_f1: number } | null;
  v2_calls: Record<string, number> | null;
  extra: { n: number; recall: number } | null;
  dial: { val_miss_target: number; cut: number; test_defects_missed: number; test_defect_accuracy: number; test_no_defect_recall: number }[] | null;
  decoys: { n: number; called_no_defect: { decoy: number; original: number }; right_class: { decoy: number; original: number } } | null;
  train: Record<"train" | "val" | "test", number>;
}

/** Architecture B on one calibrated alpha, measured on the test half (no gates for the cp_* numbers). With a
 *  no-defect class, `clear` is the share shown nothing (only "no defect" kept). */
export interface AlphaResult {
  alpha: number; accuracy: number; cp_coverage: number; cp_family_coverage: number; cp_mean_set_size: number;
  cp_single_class: number; actions: Record<"show class" | "ambiguous" | "unclear" | "re-take" | "re-aim", number> & Partial<Record<"nothing shown" | "possible defect", number>>;
  human_attention: number; single_class_wrong: number; set_threshold: number;
  clear?: number; defect_called_clear?: number; clear_right?: number;
}
export interface ArchA {
  alpha: number; temperature: number; alpha_quality: number | null; alpha_ood: number;
  cal: { crops: number; families: number; created: string | null };
  test_half: { n: number; families: number };
  ece: { raw: number; tempered: number };
  reliability: { raw: Bin[]; tempered: Bin[] };
  by_alpha: Record<string, AlphaResult>;
  silent_errors: { id: number; true: string; pred: string; confidence: number }[];
  gate_tests: { sets: { set: string; label: string; stopped: number }[]; source: string } | null;
  gates_by_source: { source: string; n: number; retake: number; reaim: number }[] | null;
}

export class ApiError extends Error {
  constructor(public status: number, message: string, public state?: string) { super(message); }
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = res.statusText, state: string | undefined;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") msg = body.detail;
      else if (body.detail?.reason) { msg = body.detail.reason; state = body.detail.state; }
    } catch { /* keep statusText */ }
    throw new ApiError(res.status, msg, state);
  }
  return res.json() as Promise<T>;
}

export const getModels = () => fetch("/api/models").then((r) => json<ModelsInfo>(r));
export const setActiveModel = (id: string) =>
  fetch("/api/models/active", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ id }) })
    .then((r) => json<ModelsInfo>(r));

export async function predict(file: File, square?: Square) {
  const fd = new FormData();
  fd.append("file", file, file.name);
  if (square) fd.append("square", `${Math.round(square.left)},${Math.round(square.top)},${Math.round(square.side)}`);
  const body = await fetch("/api/predict", { method: "POST", body: fd })
    .then((r) => json<Prediction & PredictDetail & { name: string }>(r));
  const { image, square: sq, crop, views, name: _name, ...prediction } = body;
  return { prediction: prediction as Prediction, detail: { image, square: sq, crop, views } as PredictDetail };
}

const UPLOAD_CHUNK = 64; // stay far below the server's 1000-files-per-form limit, and start streaming early

export async function batchStream(files: File[], onLine: (line: BatchLine) => void) {
  for (const part of chunkWithOffsets(files, UPLOAD_CHUNK)) {
    const fd = new FormData();
    for (const f of part.items) fd.append("files", f, f.name);
    const res = await fetch("/api/batch", { method: "POST", body: fd });
    if (!res.ok || !res.body) { await json(res); return; }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    const emit = (text: string) => { const l = JSON.parse(text) as BatchLine; onLine({ ...l, index: l.index + part.offset }); };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) emit(line);
      }
    }
    if (buf.trim()) emit(buf);
  }
}

export function predictionFromLine(l: BatchLine): Prediction {
  return {
    classes: l.classes!, probs: l.probs!, ranked: l.ranked!, top: l.top!, confidence: l.confidence!, state: l.state!,
    answer: l.answer ?? [], reason: l.reason ?? null, p_quality: l.p_quality ?? null, p_ood: l.p_ood ?? null,
    set_threshold: l.set_threshold ?? null, alpha: l.alpha ?? null, alpha_quality: l.alpha_quality ?? null,
    alpha_ood: l.alpha_ood ?? null, ms: l.ms ?? 0,
  };
}

let reportPromise: Promise<Report> | null = null;
export const getReport = (fresh = false) => {
  if (fresh || !reportPromise) reportPromise = fetch("/api/report").then((r) => json<Report>(r)).catch((e) => { reportPromise = null; throw e; });
  return reportPromise;
};
export const getSamples = () => fetch("/api/samples").then((r) => json<Sample[]>(r));
/** Inspect's one clear test crop per class (picked for clarity; Batch's samples are the random ones). */
export const getInspectSamples = () => fetch("/api/inspect-samples").then((r) => json<Sample[]>(r));
export async function sampleFile(name: string): Promise<File> {
  const r = await fetch(`/api/samples/${encodeURIComponent(name)}`);
  if (!r.ok) throw new ApiError(r.status, `sample ${name} not found`);
  const b = await r.blob();
  return new File([b], name, { type: b.type || "image/jpeg" });
}
export const cropUrl = (id: number) => `/api/crops/${id}`;
