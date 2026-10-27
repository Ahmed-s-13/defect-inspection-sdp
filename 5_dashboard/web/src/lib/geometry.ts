import type { Square } from "../api/client";

export const MIN_SIDE = 16;

export function defaultSquare(w: number, h: number): Square {
  const side = Math.min(w, h);
  return { left: Math.floor((w - side) / 2), top: Math.floor((h - side) / 2), side };
}

export function clampSquare(sq: Square, w: number, h: number, min = MIN_SIDE): Square {
  const side = Math.round(Math.max(min, Math.min(sq.side, w, h)));
  return {
    left: Math.round(Math.min(Math.max(sq.left, 0), w - side)),
    top: Math.round(Math.min(Math.max(sq.top, 0), h - side)),
    side,
  };
}
