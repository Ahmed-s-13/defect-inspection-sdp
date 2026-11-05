import type React from "react";
import { motion } from "motion/react";
import { className, pct } from "../lib/format";
import type { Item } from "../state/bench";
import { STATE_WORDS, StateGlyph, answerText } from "./StateBadge";
import { answerMisses } from "../lib/metrics";

export function Tile({ item, index, onOpen, ref }: { item: Item; index: number; onOpen(): void; ref?: React.Ref<HTMLButtonElement> }) {
  const r = item.result;
  const wrong = !!(r && item.truth && answerMisses(r, item.truth.cls)); // an answer was shown and it misses the truth
  const said = r ? (r.state === "clear" ? STATE_WORDS.clear : answerText(r) || STATE_WORDS[r.state]) : "";
  return (
    <motion.button ref={ref} layout className={`tile${wrong ? " is-wrong" : ""}`} onClick={onOpen}
      initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.96, transition: { duration: 0.18 } }}
      transition={{ duration: 0.25, delay: Math.min(index, 24) * 0.03, ease: [0.22, 1, 0.36, 1], layout: { type: "spring", stiffness: 520, damping: 42 } }}
      aria-label={`${item.name}: ${r ? said : item.status}`}>
      <motion.img layoutId={`img-${item.id}`} src={item.thumb ?? item.url} alt="" className="tile-img" />
      <span className="tile-meta">
        {r ? (
          <>
            <span className="tile-line"><StateGlyph state={r.state} size={14} /><b>{said}</b>{(r.state === "detected" || r.state === "clear" || r.state === "unrated") && <span className="fig tile-conf">{pct(r.confidence, 0)}</span>}</span>
            {item.truth && <span className={`tile-truth${wrong ? " is-wrong" : ""}`}>{wrong ? `True: ${className(item.truth.cls)}` : r.answer.length ? (r.answer.length > 1 ? "Holds the truth" : "Correct") : `True: ${className(item.truth.cls)}`}</span>}
          </>
        ) : item.status === "error" ? (
          <span className="tile-line tile-err">{item.error ?? "Could not read"}</span>
        ) : (
          <span className="tile-line tile-wait">Waiting…</span>
        )}
      </span>
    </motion.button>
  );
}
