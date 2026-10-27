import { motion, useReducedMotion, useSpring, useTransform } from "motion/react";
import { useEffect } from "react";

/** A figure that morphs from its previous value. It mounts at its value (never counts up from zero). */
export function AnimatedNumber({ value, format, className }: { value: number; format(v: number): string; className?: string }) {
  const reduce = useReducedMotion();
  const mv = useSpring(value, { stiffness: 220, damping: 30 });
  useEffect(() => { if (reduce) mv.jump(value); else mv.set(value); }, [value, reduce, mv]);
  const text = useTransform(mv, (v) => format(v));
  return <motion.span className={`fig ${className ?? ""}`}>{text}</motion.span>;
}
