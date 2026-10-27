import type { Prediction, UiState } from "../api/client";
import { className } from "../lib/format";

/** The visor's words for each of architecture A's answers. */
export const STATE_WORDS: Record<UiState, string> = {
  detected: "Defect detected", clear: "No defect", ambiguous: "Ambiguous", unclear: "Unclear", retake: "Bad image",
  cannot_assess: "Cannot assess", unrated: "Not calibrated", system_fault: "System fault",
};

/** What the inspector is asked to do, in a few words. */
export function actionLine(r: Prediction): string {
  if (r.state === "retake") return `Re-take the photo${r.reason ? `: ${r.reason}` : ""}.`;
  if (r.state === "cannot_assess") return "Re-aim: this looks unlike any training photo.";
  if (r.state === "unclear") return "No class is likely enough. Re-inspect.";
  if (r.state === "ambiguous") return `Could be ${r.answer.map((c) => className(c).toLowerCase()).join(" or ")}. A person should check.`;
  return "";
}

/** The answer itself: the class, the classes of an ambiguous set, or nothing when a gate stopped the crop. */
export function answerText(r: Prediction): string {
  if (r.state === "unrated") return className(r.top);
  if (r.state === "clear") return "Intact surface";
  return r.answer.map(className).join(" or ");
}

const COLOR: Record<UiState, string> = {
  detected: "detected", clear: "clear", ambiguous: "verify", unclear: "verify", retake: "unfamiliar", cannot_assess: "unfamiliar",
  unrated: "unrated", system_fault: "fault",
};

export function StateGlyph({ state, size = 16 }: { state: UiState; size?: number }) {
  const c = `var(--${COLOR[state]})`;
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" className="glyph">
      {state === "detected" && <><rect x="2" y="2" width="12" height="12" fill="none" stroke={c} strokeWidth="2" /><rect x="5.5" y="5.5" width="5" height="5" fill={c} /></>}
      {state === "clear" && <><rect x="2" y="2" width="12" height="12" fill="none" stroke={c} strokeWidth="2" /><path d="M4.8 8.2 7 10.4l4.3-4.6" fill="none" stroke={c} strokeWidth="2" /></>}
      {state === "ambiguous" && <><path d="M8 1.8 15 14H1z" fill={c} /><path d="M8 6v4" stroke="#16181B" strokeWidth="1.6" /><circle cx="8" cy="12" r=".9" fill="#16181B" /></>}
      {state === "unclear" && <><path d="M8 1.8 15 14H1z" fill="none" stroke={c} strokeWidth="2" /><circle cx="8" cy="10.5" r="1.1" fill={c} /></>}
      {state === "retake" && <><rect x="1.5" y="4" width="13" height="10" rx="1.5" fill="none" stroke={c} strokeWidth="2" /><path d="M5.5 4 6.8 2h2.4l1.3 2" fill="none" stroke={c} strokeWidth="1.6" /><circle cx="8" cy="9" r="2.2" fill="none" stroke={c} strokeWidth="1.6" /></>}
      {state === "cannot_assess" && <><circle cx="8" cy="8" r="6" fill="none" stroke={c} strokeWidth="2" /><path d="M3.8 12.2 12.2 3.8" stroke={c} strokeWidth="2" /></>}
      {state === "unrated" && <circle cx="8" cy="8" r="5.5" fill="none" stroke={c} strokeWidth="2" strokeDasharray="3 2" />}
      {state === "system_fault" && <><path d="M8 1.8 15 14H1z" fill="none" stroke={c} strokeWidth="2" /><path d="M8 6v4" stroke={c} strokeWidth="1.8" /><circle cx="8" cy="12" r=".9" fill={c} /></>}
    </svg>
  );
}

export function StateBadge({ state }: { state: UiState }) {
  return <span className={`state state-${state}`}><StateGlyph state={state} /> {STATE_WORDS[state]}</span>;
}
