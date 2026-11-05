import type React from "react";
import { AnimatePresence, motion } from "motion/react";
import { useRef, useState } from "react";
import { sampleUrl, type Sample, type Square, type StateId } from "../api/client";
import { className, int } from "../lib/format";
import { defaultSquare } from "../lib/geometry";
import { useElementSize } from "../lib/useElementSize";
import type { Item } from "../state/bench";
import { Reticle } from "./Reticle";
import { FanLayer } from "./ViewsFan";

interface Props {
  item: Item | null; square: Square | null; tone: StateId | null; fan: string | null;
  onClear(): void; onBack?: () => void; samples: Sample[]; onSample(s: Sample): void;
}

/** Static dash: the well shows one of the sample photos with its saved answer; there is no upload, drop or paste. */
export function ImageWell({ item, square, tone, fan, onClear, onBack, samples, onSample }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const { w, h } = useElementSize(ref);
  const [nat, setNat] = useState<{ id: string; w: number; h: number } | null>(null);

  const dims = item?.detail?.image ?? (nat && nat.id === item?.id ? { width: nat.w, height: nat.h } : null);
  const scale = dims && w && h ? Math.min(w / dims.width, h / dims.height) : 0;
  const dw = dims ? dims.width * scale : 0, dh = dims ? dims.height * scale : 0;
  const ox = (w - dw) / 2, oy = (h - dh) / 2;
  const sq = square ?? (dims ? defaultSquare(dims.width, dims.height) : null);

  const firstPerClass = samples.filter((s, i) => samples.findIndex((t) => t.cls === s.cls) === i);
  const shown = sq;
  const ease = [0.22, 1, 0.36, 1] as const;
  const I = (i: number) => ({ "--i": i }) as React.CSSProperties;

  return (
    <div className="well-col enter" style={{ "--i": 0 } as React.CSSProperties}>
      <div ref={ref} className={`well${item ? "" : " is-empty"}`}>
        <AnimatePresence mode="wait" initial={false}>
          {item ? (
            <motion.div key="photo" className="well-layer"
              initial={{ opacity: 0, scale: 0.985 }} animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.97, transition: { duration: 0.24, ease } }} transition={{ duration: 0.32, ease }}>
              <motion.img layoutId={`img-${item.id}`} className="well-image" src={item.url} alt={item.name}
                onLoad={(e) => setNat({ id: item.id, w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                style={{ left: ox, top: oy, width: dw || undefined, height: dh || undefined }} />
              {dims && sq && scale > 0 && (
                <Reticle square={sq} scale={scale} ox={ox} oy={oy} tone={tone} />
              )}
              <button className="well-close" onClick={onClear} aria-label="Remove photo" title="Remove photo (Esc)">
                <svg viewBox="0 0 16 16" aria-hidden="true"><path className="x-under" d="M3.5 3.5 12.5 12.5 M12.5 3.5 3.5 12.5" /><path d="M3.5 3.5 12.5 12.5 M12.5 3.5 3.5 12.5" /></svg>
              </button>
              {fan && sq && scale > 0 && (
                <FanLayer itemId={item.id} run={item.run} preview={fan} w={w} h={h} side={sq.side * scale}
                  cx={ox + (sq.left + sq.side / 2) * scale} cy={oy + (sq.top + sq.side / 2) * scale} />
              )}
            </motion.div>
          ) : (
            <motion.div key="empty" className="well-layer well-empty"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, transition: { duration: 0.18 } }} transition={{ duration: 0.3 }}>
              <p className="well-empty-title enter" style={I(0)}>Pick a test photo</p>
              <p className="well-empty-sub enter" style={I(1)}>One photo per class, from photo groups the model never trained on. Each shows the answer the model gave it.</p>
              {firstPerClass.length > 0 && (
                <div className="well-samples enter" style={I(2)} aria-label="Test photos">
                  {firstPerClass.map((s) => (
                    <button key={s.file} className="sample-chip" onClick={() => onSample(s)}>
                      <img src={sampleUrl(s.file)} alt="" />
                      <span>{className(s.cls)}</span>
                    </button>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      <AnimatePresence initial={false}>
        {item && (
          <motion.div key="bar" className="well-bar"
            initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6, transition: { duration: 0.2 } }} transition={{ duration: 0.3, ease }}>
            {onBack && (
              <button className="well-back" onClick={onBack}>
                <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3.5 5.5 8 10 12.5" /></svg>
                Back to batch
              </button>
            )}
            <span className="well-file" title={item.name}>{item.name}</span>
            {dims && shown && <span className="well-dims fig">{int(dims.width)} × {int(dims.height)} px · square {int(shown.side)} px at {int(shown.left)}, {int(shown.top)}</span>}
            <span className="well-actions">
              <button className="btn" onClick={onClear}>Another photo</button>
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
