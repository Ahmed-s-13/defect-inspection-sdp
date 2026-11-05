import { useEffect, useMemo, useRef, useState } from "react";
import { getInspectSamples, getSamples, sampleFile, type ModelsInfo, type Sample, type Square } from "../api/client";
import { ImageWell } from "../components/ImageWell";
import { Readout } from "../components/Readout";
import { StateGlyph } from "../components/StateBadge";
import { useCropPreview } from "../lib/useCropPreview";
import { useBench } from "../state/bench";
import { usePresent } from "../state/present";
import "../styles/inspect.css";

/** True while `active`, and for at least `ms` after it started, so a fast answer still shows the fan. */
function useHeld(active: boolean, ms: number) {
  const [held, setHeld] = useState(false);
  const started = useRef(0);
  useEffect(() => {
    if (active) { started.current = performance.now(); setHeld(true); return; }
    const t = window.setTimeout(() => setHeld(false), Math.max(0, ms - (performance.now() - started.current)));
    return () => window.clearTimeout(t);
  }, [active, ms]);
  return active || held;
}

export function InspectView({ classes, models, onBackToBatch }: { classes: string[]; models: ModelsInfo | null; onBackToBatch(): void }) {
  const bench = useBench();
  const { state } = bench;
  const item = state.inspectId ? state.items[state.inspectId] : null;
  const [samples, setSamples] = useState<Sample[]>([]);
  const [pending, setPending] = useState<{ id: string; sq: Square } | null>(null);
  // the clear one-per-class picks; the random Batch samples if those are missing (e.g. the legacy model's classes)
  useEffect(() => {
    getInspectSamples().then((s) => (s.length ? s : getSamples())).then(setSamples).catch(() => setSamples([]));
  }, [models?.active]);

  const square = (pending && pending.id === item?.id ? pending.sq : null) ?? item?.detail?.square ?? null;
  const preview = useCropPreview(item?.url, square ?? undefined);
  const present = usePresent();
  const fanning = useHeld(item?.status === "running", present.on ? 1100 : 420) && !!preview;
  const tone = !fanning && item?.status === "done" && item.result ? item.result.state : null;

  const open = (file: File, csvLabel?: string) => bench.inspectFile(file, csvLabel);
  // static dash: no pasted or dropped photos, and the square stays where the saved answer was computed

  const clear = () => { setPending(null); bench.closeInspect(); };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName);
      if (e.key === "Escape" && item && !typing) clear();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // re-binds each render so `item` is current

  const session = useMemo(() => {
    const labelled = Object.values(state.items).filter((i) => i.truth && i.result);
    return { n: labelled.length, right: labelled.filter((i) => i.truth!.cls === i.result!.top).length };
  }, [state.items]);

  const activeModel = models?.models.find((m) => m.id === models.active);
  return (
    <section className="inspect">
      <h1 className="sr-only">Inspect</h1>
      {activeModel && !activeModel.available && (
        <p className="banner-fault" role="alert"><StateGlyph state="system_fault" /> <span><b>System fault.</b> {activeModel.reason}</span></p>
      )}
      <ImageWell item={item} square={square} tone={tone === "unrated" ? null : tone} fan={fanning ? preview : null}
        onClear={clear} onBack={state.inspectFrom === "batch" ? onBackToBatch : undefined}
        samples={samples} onSample={async (s) => open(await sampleFile(s.file), s.cls)} />
      <Readout item={item} fanning={fanning} classes={classes} onTruth={(c) => item && bench.setTruth(item.id, c)} session={session} />
    </section>
  );
}
