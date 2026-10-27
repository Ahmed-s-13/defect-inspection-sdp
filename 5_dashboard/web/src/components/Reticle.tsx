import { motion } from "motion/react";
import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { Square, StateId } from "../api/client";
import { clampSquare } from "../lib/geometry";

type Mode = "move" | "nw" | "ne" | "sw" | "se";
interface Props {
  square: Square; imgW: number; imgH: number; scale: number; ox: number; oy: number;
  tone: StateId | null; onCommit(sq: Square): void; onLive?(sq: Square): void;
}

/** The exact square the model sees. Once a verdict lands it becomes the helmet's box in that state's colour.
 *  Drag to move, pull a corner bracket to resize, arrows to nudge (Shift = resize). */
export function Reticle({ square, imgW, imgH, scale, ox, oy, tone, onCommit, onLive }: Props) {
  const [sq, setSqState] = useState(square);
  const setSq = (s: Square) => { setSqState(s); onLive?.(s); };
  const start = useRef<{ mode: Mode; x: number; y: number; sq: Square } | null>(null);
  const nudge = useRef<number | undefined>(undefined);
  const direct = useRef(false); // true after the user moved it: the box sits under the pointer, no spring lag
  useEffect(() => {
    if (start.current) return;
    if (square.left !== sq.left || square.top !== sq.top || square.side !== sq.side) direct.current = false; // a programmatic move springs
    setSqState(square);
  }, [square.left, square.top, square.side]); // eslint-disable-line react-hooks/exhaustive-deps

  const down = (mode: Mode) => (e: PointerEvent) => {
    e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    start.current = { mode, x: e.clientX, y: e.clientY, sq };
    direct.current = true;
  };
  const move = (e: PointerEvent) => {
    const s = start.current;
    if (!s) return;
    const dx = (e.clientX - s.x) / scale, dy = (e.clientY - s.y) / scale;
    if (s.mode === "move") { setSq(clampSquare({ ...s.sq, left: s.sq.left + dx, top: s.sq.top + dy }, imgW, imgH)); return; }
    const sx = s.mode.includes("e") ? dx : -dx, sy = s.mode.includes("s") ? dy : -dy;
    const side = Math.max(16, Math.min(s.sq.side + Math.max(sx, sy), imgW, imgH));
    const left = s.mode.includes("w") ? s.sq.left + s.sq.side - side : s.sq.left;
    const top = s.mode.includes("n") ? s.sq.top + s.sq.side - side : s.sq.top;
    setSq(clampSquare({ left, top, side }, imgW, imgH));
  };
  const up = () => { if (start.current) { start.current = null; onCommit(sq); } };
  const key = (e: KeyboardEvent) => {
    const step = Math.max(4, Math.round(sq.side * 0.04));
    const d = ({ ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] } as Record<string, number[]>)[e.key];
    if (!d) return;
    e.preventDefault();
    direct.current = true;
    const next = e.shiftKey
      ? clampSquare({ ...sq, side: sq.side + step * (d[0] || -d[1]) }, imgW, imgH)
      : clampSquare({ ...sq, left: sq.left + d[0] * step, top: sq.top + d[1] * step }, imgW, imgH);
    setSq(next);
    window.clearTimeout(nudge.current);
    nudge.current = window.setTimeout(() => onCommit(next), 350);
  };

  return (
    <motion.div className={`reticle${tone ? ` tone-${tone}` : ""}`} role="slider" tabIndex={0}
      aria-label="Model's view: the square the model judges. Arrow keys move it, Shift with arrows resizes it."
      aria-valuetext={`left ${sq.left}, top ${sq.top}, ${sq.side} pixels square`}
      initial={false}
      style={{ width: sq.side * scale, height: sq.side * scale }}
      animate={{ x: ox + sq.left * scale, y: oy + sq.top * scale }}
      transition={direct.current ? { duration: 0 } : { type: "spring", stiffness: 700, damping: 48, mass: 0.6 }}
      onPointerDown={down("move")} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key}>
      <span className="reticle-label">Model's view <span className="fig">{sq.side} px</span></span>
      {(["nw", "ne", "sw", "se"] as Mode[]).map((m) => (
        <span key={m} className={`bracket bracket-${m}`} onPointerDown={down(m)} onPointerMove={move} onPointerUp={up} />
      ))}
    </motion.div>
  );
}
