import { AnimatePresence, LayoutGroup, MotionConfig, motion } from "motion/react";
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { getModels, type ModelsInfo } from "./api/client";
import { TopBar, TABS, type Tab } from "./components/TopBar";
import { BenchProvider, useBench } from "./state/bench";
import { PresentProvider, usePresent } from "./state/present";
import { BatchView } from "./views/BatchView";
import { DataView } from "./views/DataView";
// the Helmet page carries three.js: loaded only when its tab is opened
const HelmetView = lazy(() => import("./views/HelmetView").then((m) => ({ default: m.HelmetView })));
import { InspectView } from "./views/InspectView";
import { ModelView } from "./views/ModelView";
import { UncertaintyView } from "./views/UncertaintyView";

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
  const present = usePresent();
  const bench = useBench();

  // static dash: one saved model, so its info is read once; no switching and no presentation mode
  useEffect(() => { getModels().then(setModels).catch(() => setModels(null)); }, []);
  const classesKey = (models?.classes ?? []).join("|");
  useEffect(() => { if (models?.classes) bench.setClasses(models.classes); }, [classesKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      const n = Number(e.key);
      if (n >= 1 && n <= TABS.length) setTab(TABS[n - 1].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

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
      <TopBar tab={tab} onTab={setTab} />
      <main className="canvas">
        <LayoutGroup>
          <AnimatePresence mode="wait" initial={false} onExitComplete={onPageSwapped}>
            <motion.div key={tab} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }}
              transition={{ duration: present.timing.tab, ease: [0.22, 1, 0.36, 1] }}>
              {tab === "inspect" && <InspectView classes={classes} models={models} onBackToBatch={() => setTab("batch")} />}
              {tab === "batch" && <BatchView classes={classes} onOpen={openInInspect} />}
              {tab === "model" && <ModelView />}
              {tab === "uncertainty" && <UncertaintyView />}
              {tab === "data" && <DataView />}
              {tab === "helmet" && <Suspense fallback={null}><HelmetView /></Suspense>}
            </motion.div>
          </AnimatePresence>
        </LayoutGroup>
      </main>
    </>
  );
}
