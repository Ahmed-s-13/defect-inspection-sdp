import { AnimatePresence, LayoutGroup, MotionConfig, motion } from "motion/react";
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { getModels, getReport, setActiveModel, type ModelsInfo } from "./api/client";
import { TopBar, TABS, type Tab } from "./components/TopBar";
import { BenchProvider, useBench } from "./state/bench";
import { PresentProvider, scrollToStep, usePresent } from "./state/present";
import { BatchView } from "./views/BatchView";
import { DataView } from "./views/DataView";
// the Helmet page carries three.js: loaded only when its tab is opened
const HelmetView = lazy(() => import("./views/HelmetView").then((m) => ({ default: m.HelmetView })));
import { InspectView } from "./views/InspectView";
import { ModelView } from "./views/ModelView";

export function App() {
  return (
    <MotionConfig reducedMotion="user">
      <PresentProvider>
        <BenchProvider>
          <Shell />
        </BenchProvider>
      </PresentProvider>
    </MotionConfig>
  );
}

const typing = (t: EventTarget | null) => t instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName);

function Shell() {
  const [tab, setTab] = useState<Tab>("model"); // the Model page is the landing page
  const [models, setModels] = useState<ModelsInfo | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const present = usePresent();
  const bench = useBench();

  const refresh = useCallback(() => getModels().then((m) => { setModels(m); setModelsError(null); return m; })
    .catch((e: Error) => { setModelsError(e.message); return null; }), []);
  const benchRef = useRef(bench);
  benchRef.current = bench;
  const faulted = (m: ModelsInfo) => !m.models.find((x) => x.id === m.active)?.available;
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      const m = await refresh();
      if (alive && (!m || (!m.loaded && !faulted(m)))) setTimeout(poll, 1200); // stop once loaded or faulted
    };
    void poll();
    return () => { alive = false; };
  }, [refresh]);
  const classesKey = (models?.classes ?? []).join("|");
  useEffect(() => { if (models?.classes) bench.setClasses(models.classes); }, [classesKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const onModel = async (id: string) => {
    try {
      setModels(await setActiveModel(id));
    } catch (e) { setModelsError((e as Error).message); return; }
    getReport(true).catch(() => undefined);
    const poll = async () => {
      const m = await refresh();
      if (m && !m.loaded && !faulted(m)) { setTimeout(poll, 800); return; }
      const b = benchRef.current, cur = b.state.inspectId ? b.state.items[b.state.inspectId] : null;
      if (cur) void b.predictInspect(cur.id, cur.detail?.square); // same square as before the switch
    };
    void poll();
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "p" || e.key === "P") {
        present.toggle();
        // starting a presentation opens on its first slide (the Model page's title)
        const first = !present.on && document.querySelector<HTMLElement>('[data-step="0"]');
        if (first) requestAnimationFrame(() => scrollToStep(first));
      }
      const n = Number(e.key);
      if (n >= 1 && n <= TABS.length) setTab(TABS[n - 1].id);
      const total = document.querySelectorAll("[data-step]").length; // the Model and Data pages mark their steps
      if (present.on && total && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {
        e.preventDefault();
        const next = Math.min(Math.max(present.step + (e.key === "ArrowRight" ? 1 : -1), 0), total - 1);
        present.setStep(next);
        const el = document.querySelector<HTMLElement>(`[data-step="${next}"]`);
        if (el) scrollToStep(el);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [present, tab]);

  const { setStep } = present;
  useEffect(() => setStep(0), [tab, setStep]); // each page's presentation starts at its first section

  const tabRef = useRef(tab);
  tabRef.current = tab;
  // after the old page has left (so Batch can't record the jump as its place), new pages start at the top;
  // Batch restores its own place
  const onPageSwapped = () => { if (tabRef.current !== "batch") window.scrollTo(0, 0); };
  const openInInspect = (id: string) => { bench.openInInspect(id); setTab("inspect"); };
  const classes = models?.classes ?? [];

  return (
    <>
      <TopBar tab={tab} onTab={setTab} models={models} modelsError={modelsError} onModel={onModel} />
      <main className="canvas">
        <LayoutGroup>
          <AnimatePresence mode="wait" initial={false} onExitComplete={onPageSwapped}>
            <motion.div key={tab} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }}
              transition={{ duration: present.timing.tab, ease: [0.22, 1, 0.36, 1] }}>
              {tab === "inspect" && <InspectView classes={classes} models={models} onBackToBatch={() => setTab("batch")} />}
              {tab === "batch" && <BatchView classes={classes} onOpen={openInInspect} />}
              {tab === "model" && <ModelView />}
              {tab === "data" && <DataView />}
              {tab === "helmet" && <Suspense fallback={null}><HelmetView /></Suspense>}
            </motion.div>
          </AnimatePresence>
        </LayoutGroup>
      </main>
    </>
  );
}
