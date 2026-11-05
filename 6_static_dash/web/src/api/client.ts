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
  classes: string[];
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

/* Static dash: there is no server. Every answer below was saved from the real dashboard's server by
   export_static_data.py, into public/static-data/, and is read back here; nothing runs the model. */
const BASE = import.meta.env.BASE_URL;                  // "./": the site works from any folder it is hosted in
/** A path inside the built site, relative to its root (e.g. "static-data/report.json"). */
export const asset = (path: string) => BASE + path;
const once = <T,>(path: string) => {
  let p: Promise<T> | null = null;
  return () => (p ??= fetch(asset(path)).then((r) => json<T>(r)).catch((e) => { p = null; throw e; }));
};
interface Saved { predict: Prediction & PredictDetail & { name: string }; batch: BatchLine }
const saved = once<Record<string, Saved>>("static-data/predictions.json");
/** The saved answer for one of the sample photos; static dash offers no others. */
async function savedFor(name: string) {
  const s = (await saved())[name];
  if (!s) throw new ApiError(404, "Only the sample photos can be shown here.");
  return s;
}
const withAssets = <T extends { crop?: string; views?: string[]; thumb?: string }>(o: T): T => ({
  ...o, ...(o.crop && { crop: asset(o.crop) }), ...(o.views && { views: o.views.map(asset) }), ...(o.thumb && { thumb: asset(o.thumb) }),
});

export const getModels = once<ModelsInfo>("static-data/models.json");
/** Static dash serves one model, so there is nothing to switch to. */
export const setActiveModel = (_id: string) => getModels();

/** The saved answer for a sample at its default square (static dash's square is fixed). */
export async function predict(file: File, _square?: Square) {
  const body = withAssets((await savedFor(file.name)).predict);
  const { image, square: sq, crop, views, name: _name, ...prediction } = body;
  return { prediction: prediction as Prediction, detail: { image, square: sq, crop, views } as PredictDetail };
}

/** The saved Batch answers, handed over one by one like the server's stream. */
export async function batchStream(files: File[], onLine: (line: BatchLine) => void) {
  for (const [index, f] of files.entries()) {
    await new Promise((r) => setTimeout(r, 45));
    try {
      onLine({ ...withAssets((await savedFor(f.name)).batch), index, name: f.name });
    } catch (e) {
      onLine({ index, name: f.name, ok: false, error: (e as Error).message });
    }
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

const report = once<Report>("static-data/report.json");
export const getReport = (_fresh = false) => report();
export const getSamples = once<Sample[]>("static-data/samples.json");
/** Inspect's one clear test crop per class (picked for clarity; Batch's samples are the random ones). */
export const getInspectSamples = once<Sample[]>("static-data/inspect_samples.json");
export const sampleUrl = (name: string) => asset(`static-data/samples/${encodeURIComponent(name)}`);
export async function sampleFile(name: string): Promise<File> {
  const r = await fetch(sampleUrl(name));
  if (!r.ok) throw new ApiError(r.status, `sample ${name} not found`);
  const b = await r.blob();
  return new File([b], name, { type: b.type || "image/jpeg" });
}
export const cropUrl = (id: number) => asset(`static-data/crops/${id}.jpg`);
