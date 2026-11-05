import { useEffect, useRef } from "react";
import logoMask from "../data/logoMask.json";
import { watchNear } from "../lib/near";
import "../styles/title.css";

/** The Model page's title: a hand-drawn sketch of the team's helmet beside S.A.Q.R. (SDP'26 under it), and the SAQR
 *  logo in the top right. Both draw themselves on arrival: the logo is revealed along its own outlines (a mask traced
 *  by scripts/logo_mask.py), then filled in, so the image itself is shown unchanged. Scrolling down un-draws them
 *  (details first, the outlines last) and scrolling back up redraws them.
 *  With `step`, it is that presentation step (the first slide), shown fully drawn at the top of its run. */
export function HelmetTitle({ step }: { step?: number }) {
  const run = useRef<HTMLDivElement>(null);
  const strokes = useRef<SVGGElement>(null);
  const type = useRef<HTMLDivElement>(null);
  const logo = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const els = Array.from(strokes.current!.children) as SVGElement[];
    const marks = Array.from(logo.current?.querySelectorAll<SVGElement>(".logo-line") ?? []);
    [...els, ...marks].forEach((el) => { el.setAttribute("pathLength", "1"); el.style.strokeDasharray = "1 1"; });
    const fill = logo.current?.querySelector<SVGElement>(".logo-fill");
    const INTRO_MS = 1800, t0 = performance.now();
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let intro = reduce ? 1 : 0, raf = 0;

    // static dash: smooth on phones. The scroll position is worked out from measurements taken once (and again on
    // resize), so a frame never forces the browser to lay the page out; each stroke is written only when its value
    // changes; and nothing runs once the title is far off screen
    const last = new Map<Element, Record<string, string>>();
    const put = (el: HTMLElement | SVGElement, prop: "strokeDashoffset" | "visibility" | "opacity" | "transform", v: string) => {
      const seen = last.get(el) ?? {};
      if (seen[prop] === v) return;
      seen[prop] = v; last.set(el, seen);
      el.style[prop] = v;
    };
    let geo: { top: number; span: number; pin: number } | null = null;
    const measure = () => {
      const box = run.current, stick = box?.firstElementChild as HTMLElement | null;
      if (!box || !stick) return;
      geo = { top: box.getBoundingClientRect().top + scrollY, span: Math.max(1, box.offsetHeight - stick.offsetHeight), pin: parseFloat(getComputedStyle(stick).top) || 0 };
    };
    let prevErase = -1, prevIntro = -1;

    const paint = (now: number) => {
      raf = 0;
      if (intro < 1) intro = Math.min(1, (now - t0) / INTRO_MS);
      if (!geo) measure();
      if (!geo) return;
      const slid = Math.min(geo.span, Math.max(0, scrollY + geo.pin - geo.top));   // how far the pinned stage has moved
      const erase = Math.round(Math.min(1, slid / (geo.span * 0.85)) * 500) / 500;
      if (erase === prevErase && intro === prevIntro) { if (intro < 1) raf = requestAnimationFrame(paint); return; }
      prevErase = erase; prevIntro = intro;
      // each list is drawn in order and un-drawn in reverse: the helmet's strokes, and the logo's beside them
      const sketch = (list: SVGElement[]) => list.forEach((el, i) => {
        const k = list.length, drawn = Math.min(1, Math.max(0, (intro - (i / k) * 0.75) / 0.25));
        const kept = 1 - Math.min(1, Math.max(0, (erase - ((k - 1 - i) / k) * 0.75) / 0.25));
        const shown = Math.min(drawn, kept);
        put(el, "strokeDashoffset", (1 - shown).toFixed(3));
        put(el, "visibility", shown <= 0.001 ? "hidden" : "visible"); // a round cap would leave a dot
      });
      sketch(els);
      sketch(marks);
      // the rest of the logo fills in once its outlines are drawn, and is the first to go
      if (fill) put(fill, "opacity", Math.min(Math.min(1, Math.max(0, (intro - 0.75) / 0.25)), 1 - Math.min(1, Math.max(0, erase / 0.3))).toFixed(3));
      if (type.current) {
        put(type.current, "opacity", (1 - erase * 0.85).toFixed(3));
        put(type.current, "transform", `translateY(${(-erase * 40).toFixed(1)}px)`);
      }
      if (intro < 1) raf = requestAnimationFrame(paint);
    };
    const near = watchNear(run.current!, () => { if (!raf) raf = requestAnimationFrame(paint); });
    const onScroll = () => { if (near.near && !raf) raf = requestAnimationFrame(paint); };
    const onResize = () => { geo = null; onScroll(); };
    raf = requestAnimationFrame(paint);
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onResize);
    void document.fonts?.ready.then(onResize);
    return () => { near.stop(); cancelAnimationFrame(raf); window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onResize); };
  }, []);

  return (
    <div className="title-run" ref={run}>
      <div className="title-stage">
        <svg className="contours" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <path d="M-50,520 C180,430 300,610 520,520 C700,450 820,600 1250,470" />
      <path d="M-50,560 C170,480 320,660 540,565 C720,495 840,650 1250,520" />
      <path d="M-50,600 C160,530 340,700 560,610 C740,540 860,700 1250,570" />
      <path d="M-50,170 C210,250 380,90 600,160 C740,205 820,120 880,-40" /> {/* the top two lines rise out of the page before the logo */}
      <path d="M-50,130 C220,210 390,50 610,120 C740,162 800,80 850,-60" />
      <path d="M760,420 C820,300 990,300 1040,420 C1090,540 900,610 820,560 C780,535 745,480 760,420 Z" />
      <path d="M800,425 C845,345 965,345 1000,425 C1035,505 905,555 850,520 C820,500 790,470 800,425 Z" />
    </svg>
        <svg className="sketch" viewBox="90 80 640 480" role="img" aria-label="Hand-drawn sketch of the team's inspection helmet: hard hat with front light and camera, and a near-eye display">
          <defs>
            <filter id="wobble" x="-5%" y="-5%" width="110%" height="110%">
              <feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves={2} seed={7} result="n" />
              <feDisplacementMap in="SourceGraphic" in2="n" scale={4.2} xChannelSelector="R" yChannelSelector="G" />
            </filter>
          </defs>
          <g filter="url(#wobble)" ref={strokes}>
        {/* shell: forehead, crown, back */}
        <path className="ink"   d="M238,318 C232,250 262,180 322,145 C380,112 480,100 556,118 C640,138 690,210 694,290 C696,320 690,342 680,356" />
        <path className="ghost" d="M242,314 C237,251 266,184 325,149 C382,116 479,105 554,122 C636,142 685,212 689,290 C691,318 686,338 677,352" />
        {/* peak: where it leaves the shell, its tip, and the long lower rim of the hat */}
        <path className="ink"   d="M166,345 C196,330 220,318 240,310" />
        <path className="ink"   d="M166,345 C157,350 160,359 173,363 C222,380 272,391 340,400 C430,412 540,404 610,386 C650,375 672,365 680,356" />
        <path className="ink"   d="M176,350 C219,366 271,378 345,387 C432,398 538,392 606,374 C645,364 667,354 676,346" />
        <path className="ghost" d="M170,368 C220,384 268,395 342,404 C430,416 538,409 608,391" />
        {/* crown ridge (two edges) and a side contour */}
        <path className="ink"   d="M262,214 C300,158 380,122 470,116 C560,112 630,150 668,214" />
        <path className="ink"   d="M276,226 C312,172 388,138 472,132 C556,129 622,164 658,226" />
        <path className="ghost" d="M300,300 C340,250 430,226 520,232 C590,238 640,268 670,306" />
        {/* vents at the back of the crown */}
        <path className="ink" d="M600,168 L622,184" /><path className="ink" d="M610,186 L631,203" /><path className="ink" d="M617,205 L637,223" />
        {/* camera box seated on the forehead: front, top, side */}
        <path transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="ink"   d="M230,226 L280,219 Q286,218 286,225 L289,280 Q289,286 283,287 L236,291 Q230,292 230,286 Z" />
        <path transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="ink"   d="M230,226 L256,207 L308,201 L284,219" />
        <path transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="ink"   d="M308,201 L312,262 L289,283" />
        <path transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="hatch" d="M292,222 L304,212 M293,240 L307,229 M294,258 L308,247 M295,274 L306,265" />
        <circle transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="lens" cx="258" cy="254" r="19" /><circle transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="lens" cx="258" cy="254" r="9" />
        <path transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="hatch" d="M252,247 Q256,244 261,246" />
        {/* flashlight: barrel into the box, lens rings, beam */}
        <path transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="ink"   d="M180,226 L230,230 M180,287 L232,288" />
        <ellipse transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="ink"  cx="178" cy="256" rx="22" ry="31" />
        <ellipse transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="lens" cx="175" cy="256" rx="14" ry="20" />
        <ellipse transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="lens" cx="173" cy="256" rx="6" ry="9" />
        <path transform="matrix(1.2 0 0 1.2 -51 -29)" vectorEffect="non-scaling-stroke" className="hatch" d="M142,242 L118,234 M140,257 L112,257 M142,272 L118,281" />
        {/* near-eye display: side mount, arm, prism */}
        <path className="ink"   d="M304,350 L336,346 Q342,346 342,352 L343,382 Q343,388 337,388 L307,391 Q301,391 301,385 L300,356 Q300,350 304,350 Z" />
        <path className="ink"   d="M306,390 C298,400 290,408 282,418" />
        <path className="ink"   d="M226,420 L288,410 L296,462 L238,476 Z" />
        <path className="ghost" d="M235,428 L283,420 L289,457 L244,467 Z" />
        <path className="hatch" d="M250,462 L282,430 M262,466 L288,440" />
        {/* ratchet knob, chin strap and buckle */}
        <circle className="ink"  cx="600" cy="392" r="24" /><circle className="lens" cx="600" cy="392" r="11" />
        <path className="ink"   d="M318,392 C330,440 380,490 452,512" />
        <path className="ink"   d="M332,390 C344,436 392,482 456,503" />
        <path className="ink"   d="M596,416 C580,462 540,500 490,512" />
        <path className="ink"   d="M610,412 C594,462 552,508 492,526" />
        <path className="ink"   d="M452,505 L489,500 L493,526 L456,531 Z" />
        {/* suspension band seen under the rim, and shade under the peak */}
        <path className="ghost" d="M360,402 C420,420 520,422 590,404" />
        <path className="hatch" d="M184,364 L198,352 M202,369 L216,357 M220,373 L233,362" />
        {/* the detection mark: the only colour, last to draw, first to go */}
        <path className="mark"  d="M430,232 L430,218 L444,218 M492,218 L506,218 L506,232 M506,274 L506,288 L492,288 M444,288 L430,288 L430,274" />
          </g>
        </svg>
        <div className="title-type" ref={type}>
          <h1 className="saqr">S.A.Q.R.</h1>
          <p className="sdp">SDP’26</p>
        </div>
        {/* the SAQR logo from the brand identity, mirrored to face left (web/public/brand/saqr_logo.png); the mask shows it
            along its outlines first */}
        <svg className="saqr-logo" ref={logo} viewBox={`0 0 ${logoMask.w} ${logoMask.h}`} aria-hidden="true">
          <defs>
            <mask id="saqr-logo-mask" maskUnits="userSpaceOnUse" x="0" y="0" width={logoMask.w} height={logoMask.h}>
              <rect className="logo-fill" width={logoMask.w} height={logoMask.h} fill="#fff" />
              {logoMask.paths.map((d) => <path key={d.slice(0, 24)} className="logo-line" d={d} fill="none" stroke="#fff" strokeWidth={logoMask.pen} strokeLinecap="round" strokeLinejoin="round" />)}
            </mask>
            {/* the logo in the page's secondary ink (as SDP'26): its shape and shading (the image's alpha) filled with
                var(--ink-2), the alpha scaled so the darkest parts come out as the full colour */}
            <filter id="saqr-logo-grey" x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
              <feComponentTransfer in="SourceAlpha" result="a"><feFuncA type="linear" slope={1 / logoMask.alphaMax} /></feComponentTransfer>
              <feFlood style={{ floodColor: "var(--ink-2)" }} />
              <feComposite in2="a" operator="in" />
            </filter>
          </defs>
          {/* static dash: the filter's result baked into an image (brand/saqr_logo_ink2.png), so a phone doesn't run the
              filter again on every frame the mask changes */}
          <image href="brand/saqr_logo_ink2.png" width={logoMask.w} height={logoMask.h} mask="url(#saqr-logo-mask)" />
        </svg>
      </div>
      {step != null && <i hidden data-step={step} data-step-at={0} />}
    </div>
  );
}
