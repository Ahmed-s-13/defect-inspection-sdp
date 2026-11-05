import { motion, useReducedMotion, useScroll } from "motion/react";
import { useRef } from "react";

/* static dash: where the browser can tie an animation to the scroll itself (CSS scroll-driven animations), the rule
   is drawn that way, off the main thread, instead of by a per-frame scroll tracker; elsewhere, as before. */
const CSS_SCROLL = typeof CSS !== "undefined" && CSS.supports?.("animation-timeline: view()");

/** A section's top rule, drawn left to right by the scroll itself (like the title sketch's lines). */
export const RuleDraw = CSS_SCROLL ? RuleCss : RuleMotion;

function RuleCss() {
  return <div className="rule-draw is-scroll-driven" aria-hidden="true" />;
}

function RuleMotion() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 95%", "start 55%"] });
  const reduce = useReducedMotion();
  return <motion.div ref={ref} className="rule-draw" aria-hidden="true" style={{ scaleX: reduce ? 1 : scrollYProgress }} />;
}
