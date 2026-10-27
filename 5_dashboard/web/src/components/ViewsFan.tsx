import { motion } from "motion/react";
import { usePresent } from "../state/present";

/** CSS for view k, in the same order as the model's dihedral_t: 4 quarter turns, then the same flipped. */
export const cssView = (k: number) => `${k >= 4 ? "scaleX(-1) " : ""}rotate(${-90 * (k % 4)}deg)`;
export const viewId = (itemId: string, run: number, k: number) => `view-${itemId}-${run}-${k}`;

/** While the model runs, the crop fans out of the reticle into the 8 views it averages (drawn in the well). */
export function FanLayer({ itemId, run, preview, cx, cy, side, w, h }: {
  itemId: string; run: number; preview: string; cx: number; cy: number; side: number; w: number; h: number;
}) {
  const { timing } = usePresent();
  const tile = Math.max(44, Math.min(side * 0.42, 104));
  const r = side / 2 + tile * 0.62;
  return (
    <div className="fan-layer" aria-hidden="true">
      {Array.from({ length: 8 }, (_, k) => {
        const a = (k / 8) * Math.PI * 2 - Math.PI / 2;
        const x = Math.min(Math.max(cx + Math.cos(a) * r, tile / 2 + 6), w - tile / 2 - 6);
        const y = Math.min(Math.max(cy + Math.sin(a) * r, tile / 2 + 6), h - tile / 2 - 6);
        return (
          <motion.div key={k} layoutId={viewId(itemId, run, k)} className="fan-view"
            style={{ left: x - tile / 2, top: y - tile / 2, width: tile, height: tile }}
            initial={{ x: cx - x, y: cy - y, scale: 0.35, opacity: 0 }} animate={{ x: 0, y: 0, scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: timing.fanStiffness, damping: timing.fanDamping, delay: k * timing.stagger }}>
            <img src={preview} alt="" style={{ transform: cssView(k) }} />
          </motion.div>
        );
      })}
    </div>
  );
}

/** When the answer lands, the 8 views fold out of the well into this strip (shared layout ids). */
export function ViewsStrip({ itemId, run, views, fanning }: { itemId: string; run: number; views: string[] | null; fanning: boolean }) {
  const { timing } = usePresent();
  return (
    <figure className="views">
      <div className="views-stage">
        {!fanning && views && views.map((src, k) => (
          <motion.div key={k} layoutId={viewId(itemId, run, k)} className="view" style={{ left: `${(k + 0.5) * 12.5}%` }}
            transition={{ type: "spring", stiffness: timing.fanStiffness, damping: timing.fanDamping, delay: k * timing.stagger * 0.6 }}>
            <img src={src} alt="" />
          </motion.div>
        ))}
      </div>
      <figcaption>{fanning ? "Looking at the patch 8 ways…" : "Seen 8 ways, rotated and flipped. The model averages what it sees."}</figcaption>
    </figure>
  );
}
