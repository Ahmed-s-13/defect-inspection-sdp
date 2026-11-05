import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export interface Timing { fanStiffness: number; fanDamping: number; stagger: number; tab: number }
const DAILY: Timing = { fanStiffness: 260, fanDamping: 28, stagger: 0.03, tab: 0.18 };
const SHOW: Timing = { fanStiffness: 110, fanDamping: 18, stagger: 0.07, tab: 0.32 };

interface Present { on: boolean; toggle(): void; step: number; setStep(n: number): void; timing: Timing }
const Ctx = createContext<Present>({ on: false, toggle() {}, step: 0, setStep() {}, timing: DAILY });

export function PresentProvider({ children }: { children: ReactNode }) {
  const [on, setOn] = useState(false);
  const [step, setStep] = useState(0);
  useEffect(() => { document.documentElement.dataset.present = on ? "on" : "off"; }, [on]);
  const value = useMemo(() => ({ on, toggle: () => { setOn((v) => !v); setStep(0); }, step, setStep, timing: on ? SHOW : DAILY }), [on, step]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
export const usePresent = () => useContext(Ctx);

/** Brings a presentation step into view. A step marked `data-step-at` sits in a pinned scroll scene (its parent
 *  holds the sticky stage: a `.scene`, or the title's `.title-stage`): it scrolls to that fraction of the way
 *  through the scene. */
export function scrollToStep(el: HTMLElement) {
  const at = el.dataset.stepAt, run = el.parentElement, stage = run?.querySelector<HTMLElement>(".scene, .title-stage");
  if (at === undefined || !run || !stage) { el.scrollIntoView({ behavior: "smooth", block: "center" }); return; }
  const stick = parseFloat(getComputedStyle(stage).top) || 0;
  const top = run.getBoundingClientRect().top + scrollY - stick + Number(at) * (run.offsetHeight - stage.offsetHeight);
  scrollTo({ top, behavior: "smooth" });
}
