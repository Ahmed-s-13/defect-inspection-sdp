import { motion, useReducedMotion, useScroll } from "motion/react";
import { useRef } from "react";

/** A section's top rule, drawn left to right by the scroll itself (like the title sketch's lines). */
export function RuleDraw() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 95%", "start 55%"] });
  const reduce = useReducedMotion();
  return <motion.div ref={ref} className="rule-draw" aria-hidden="true" style={{ scaleX: reduce ? 1 : scrollYProgress }} />;
}
