import { useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { usePresent } from "../state/present";
import { watchNear } from "./near";
import { nextShown } from "./reveal";

/** A section's entrance and exit, tied to the scroll direction like the title sketch.
 *  Scrolling down, it plays in once its top rises above `show` (a fraction of the viewport height).
 *  Scrolling up, it dissolves once its top drops below `hide`, while it is still on screen to be seen,
 *  so scrolling down again replays it. Presentation mode slows it; reduced motion shows it at once.
 *  `t(s)` scales a duration or delay in seconds. */
export function useReveal<T extends Element>(show = 0.85, hide = 0.55) {
  const ref = useRef<T>(null);
  const [reached, setReached] = useState(false);
  const reduce = !!useReducedMotion();
  const { on } = usePresent();

  useEffect(() => {
    let raf = 0, lastY = scrollY;
    const check = () => {
      raf = 0;
      const el = ref.current;
      if (!el) return;
      const top = el.getBoundingClientRect().top, dir = scrollY - lastY;
      lastY = scrollY;
      setReached((prev) => nextShown(prev, top, dir, innerHeight, show, hide));
    };
    // static dash: a section far off screen is not checked on every scroll (its state only changes on screen)
    const near = ref.current ? watchNear(ref.current, () => { lastY = scrollY; if (!raf) raf = requestAnimationFrame(check); }, "50% 0px") : null;
    const onScroll = () => { if ((!near || near.near) && !raf) raf = requestAnimationFrame(check); };
    check();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { near?.stop(); cancelAnimationFrame(raf); window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, [show, hide]);

  const shown = reduce || reached;
  const k = (on ? 1.8 : 1) * (shown ? 1 : 0.4);
  return { ref, shown, t: (s: number) => (reduce ? 0 : s * k) };
}
export type Reveal = { shown: boolean; t: (s: number) => number };
