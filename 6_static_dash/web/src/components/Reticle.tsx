import { motion } from "motion/react";
import type { Square, StateId } from "../api/client";

interface Props { square: Square; scale: number; ox: number; oy: number; tone: StateId | null }

/** The exact square the model saw. Once the verdict lands it becomes the helmet's box in that state's colour.
 *  Static dash: the square is fixed at the sample's default, the one its saved answer was computed for. */
export function Reticle({ square: sq, scale, ox, oy, tone }: Props) {
  return (
    <motion.div className={`reticle is-fixed${tone ? ` tone-${tone}` : ""}`} aria-hidden="true" initial={false}
      style={{ width: sq.side * scale, height: sq.side * scale, cursor: "default" }}
      animate={{ x: ox + sq.left * scale, y: oy + sq.top * scale }} transition={{ type: "spring", stiffness: 700, damping: 48, mass: 0.6 }}>
      <span className="reticle-label">Model's view <span className="fig">{sq.side} px</span></span>
      {["nw", "ne", "sw", "se"].map((m) => <span key={m} className={`bracket bracket-${m}`} />)}
    </motion.div>
  );
}
