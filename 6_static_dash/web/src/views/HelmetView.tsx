import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { HELMET_PARTS, HELMET_SUMMARY } from "../data/helmet";
import { HelmetScene } from "../lib/helmetScene";
import "../styles/helmet.css";

const ease = [0.22, 1, 0.36, 1] as const;
const num = (i: number) => String(i + 1).padStart(2, "0");

/** The Helmet page: SAQR Prototype 1 in 3D, turning slowly over big type. Drag it to turn it by hand; hover a
 *  component (on the helmet or in the list) to light it, click it to fly in and read what it is. */
export function HelmetView() {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<HelmetScene | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");
  const [hovered, setHovered] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => {
    let s: HelmetScene;
    try {
      s = new HelmetScene(host.current!, {
        onLoad: () => setState("ready"),
        onError: (m) => { setError(m); setState("error"); },
        onHover: setHovered,
        onPick: (id) => setPicked(id),
      });
    } catch (e) { // no WebGL
      setError((e as Error).message);
      setState("error");
      return;
    }
    scene.current = s;
    return () => { s.dispose(); scene.current = null; };
  }, []);
  useEffect(() => { scene.current?.pick(picked); }, [picked]);
  // static dash, phones: the list sits under the helmet, so picking a part brings the helmet and its card into view
  useEffect(() => {
    if (!picked || !matchMedia("(max-width: 860px)").matches || !host.current) return;
    const bar = document.querySelector<HTMLElement>(".topbar")?.offsetHeight ?? 0;
    window.scrollTo({ top: window.scrollY + host.current.getBoundingClientRect().top - bar, behavior: "smooth" });
  }, [picked]);
  const touch = typeof matchMedia !== "undefined" && matchMedia("(pointer: coarse)").matches;

  // Esc closes the card; the arrow keys step through the components while one is open
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPicked(null);
      if (!picked || (e.key !== "ArrowRight" && e.key !== "ArrowLeft")) return;
      e.preventDefault();
      const i = HELMET_PARTS.findIndex((p) => p.id === picked), n = HELMET_PARTS.length;
      setPicked(HELMET_PARTS[(i + (e.key === "ArrowRight" ? 1 : n - 1)) % n].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [picked]);

  const index = picked ? HELMET_PARTS.findIndex((p) => p.id === picked) : -1;
  const part = index >= 0 ? HELMET_PARTS[index] : null;
  const step = (d: number) => setPicked(HELMET_PARTS[(index + d + HELMET_PARTS.length) % HELMET_PARTS.length].id);
  return (
    <section className={`helmet-page${picked ? " is-picked" : ""}`} aria-label="SAQR Prototype 1 in 3D">
      <svg className="helmet-contours" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
        <path d="M-50,520 C180,430 300,610 520,520 C700,450 820,600 1250,470" />
        <path d="M-50,560 C170,480 320,660 540,565 C720,495 840,650 1250,520" />
        <path d="M-50,600 C160,530 340,700 560,610 C740,540 860,700 1250,570" />
        <path d="M-50,170 C210,250 380,90 600,160 C820,230 930,80 1250,180" />
        <path d="M-50,130 C220,210 390,50 610,120 C830,190 940,40 1250,140" />
        <path d="M760,420 C820,300 990,300 1040,420 C1090,540 900,610 820,560 C780,535 745,480 760,420 Z" />
        <path d="M800,425 C845,345 965,345 1000,425 C1035,505 905,555 850,520 C820,500 790,470 800,425 Z" />
      </svg>
      <div className="helmet-type" aria-hidden="true"><span>S.A.Q.R.</span></div>

      <div className="helmet-stage" ref={host} />

      <header className="helmet-head">
        <p className="helmet-kicker">The helmet</p>
        <h1 className="helmet-title">Prototype <span>1</span></h1>
        <p className="helmet-hint">{state === "loading" ? "Loading the model…" : state === "error" ? `The 3D model could not be shown: ${error}` : touch ? "Drag sideways to turn it. Tap a part to look closer." : "Hold and drag to turn it. Click a part to look closer."}</p>
      </header>

      <nav className="helmet-index" aria-label="Components">
        <ol>
          {HELMET_PARTS.map((p, i) => (
            <li key={p.id}>
              <button className={`${hovered === p.id ? "is-hover" : ""}${picked === p.id ? " is-on" : ""}`} aria-pressed={picked === p.id}
                onMouseEnter={() => scene.current?.hover(p.id)} onMouseLeave={() => scene.current?.hover(null)}
                onFocus={() => scene.current?.hover(p.id)} onBlur={() => scene.current?.hover(null)}
                onClick={() => setPicked(picked === p.id ? null : p.id)}>
                <span className="fig">{num(i)}</span>{p.name}
              </button>
            </li>
          ))}
        </ol>
      </nav>

      <AnimatePresence>
        {!picked && (
          <motion.p key="summary" className="helmet-summary" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }} transition={{ duration: 0.35, ease }}>
            {HELMET_SUMMARY.join(" ")}
          </motion.p>
        )}
      </AnimatePresence>

      <AnimatePresence mode="wait">
        {part && (
          <motion.aside key={part.id} className="helmet-card" role="dialog" aria-label={part.name}
            initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10, transition: { duration: 0.18 } }}
            transition={{ duration: 0.45, ease, delay: 0.25 }}>
            <p className="helmet-card-n"><span className="fig">{num(index)}</span> of {HELMET_PARTS.length}</p>
            <h2>{part.name}</h2>
            {part.text.map((t) => <p key={t}>{t}</p>)}
            <div className="helmet-card-nav">
              <button onClick={() => step(-1)} aria-label="Previous component"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7" /></svg></button>
              <button onClick={() => step(1)} aria-label="Next component"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7" /></svg></button>
              <button className="helmet-card-close" onClick={() => setPicked(null)}>Back to the helmet <kbd>Esc</kbd></button>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </section>
  );
}

