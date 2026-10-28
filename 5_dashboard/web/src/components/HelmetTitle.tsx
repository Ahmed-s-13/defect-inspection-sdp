import { useEffect, useRef } from "react";
import "../styles/title.css";

/** The Model page's title: a hand-drawn sketch of the team's helmet beside S.A.Q.R. (SDP'26 under it), and a drawn
 *  falcon in the top right. The sketches draw themselves on arrival. Scrolling down un-draws the helmet (details first,
 *  the shell last) while the falcon flaps and dives across the page, gone by the end of the title; scrolling back up
 *  reverses both.
 *  With `step`, it is that presentation step (the first slide), shown fully drawn at the top of its run. */
export function HelmetTitle({ step }: { step?: number }) {
  const run = useRef<HTMLDivElement>(null);
  const strokes = useRef<SVGGElement>(null);
  const type = useRef<HTMLDivElement>(null);
  const falcon = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const els = Array.from(strokes.current!.children) as SVGElement[];
    els.forEach((el) => { el.setAttribute("pathLength", "1"); el.style.strokeDasharray = "1 1"; });
    const wings = Array.from(falcon.current?.querySelectorAll<SVGGElement>(".falcon-wing") ?? []);
    const n = els.length, INTRO_MS = 1800, t0 = performance.now();
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let intro = reduce ? 1 : 0, raf = 0;

    const paint = (now: number) => {
      raf = 0;
      if (intro < 1) intro = Math.min(1, (now - t0) / INTRO_MS);
      const box = run.current;
      if (!box) return;
      const stick = box.firstElementChild as HTMLElement;
      const span = Math.max(1, box.offsetHeight - stick.offsetHeight);
      const erase = Math.min(1, Math.max(0, (stick.getBoundingClientRect().top - box.getBoundingClientRect().top) / (span * 0.85)));
      els.forEach((el, i) => {
        const drawn = Math.min(1, Math.max(0, (intro - (i / n) * 0.75) / 0.25));
        const kept = 1 - Math.min(1, Math.max(0, (erase - ((n - 1 - i) / n) * 0.75) / 0.25));
        const shown = Math.min(drawn, kept);
        el.style.strokeDashoffset = String(1 - shown);
        el.style.visibility = shown <= 0.001 ? "hidden" : "visible"; // a round cap would leave a dot
      });
      if (type.current) {
        type.current.style.opacity = String(1 - erase * 0.85);
        type.current.style.transform = `translateY(${-erase * 40}px)`;
      }
      const f = falcon.current;
      if (f) {
        // the falcon glides in from the right while the helmet draws, then the scroll flies it left above the
        // helmet, beating its wing as it goes, faded out by the end of the title run
        const arrive = 1 - Math.pow(1 - Math.min(1, Math.max(0, (intro - 0.3) / 0.6)), 3);
        const W = stick.offsetWidth, H = stick.offsetHeight, e = erase;
        // across the band above the helmet: a gentle rise, then a glide down as it leaves
        const x = -e * W * 1.05, y = -Math.sin(e * Math.PI) * 24 + e * e * 40;
        const tilt = Math.atan2(-Math.cos(e * Math.PI) * 24 * Math.PI + 80 * e, 1.05 * W) * (180 / Math.PI); // nose follows the path
        f.style.transform = `translate(${x + (1 - arrive) * 60}px, ${y}px) rotate(${-tilt}deg) scale(${1 - e * 0.3})`;
        f.style.opacity = String(arrive * (1 - Math.min(1, Math.max(0, (e - 0.72) / 0.28))));
        const flap = reduce ? 0 : 13 * Math.sin(e * Math.PI * 9);      // the wing beats up and down about the shoulder
        wings.forEach((w) => { w.style.transform = `rotate(${flap}deg)`; });
      }
      if (intro < 1) raf = requestAnimationFrame(paint);
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(paint); };
    raf = requestAnimationFrame(paint);
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, []);

  return (
    <div className="title-run" ref={run}>
      <div className="title-stage">
        <svg className="contours" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <path d="M-50,520 C180,430 300,610 520,520 C700,450 820,600 1250,470" />
      <path d="M-50,560 C170,480 320,660 540,565 C720,495 840,650 1250,520" />
      <path d="M-50,600 C160,530 340,700 560,610 C740,540 860,700 1250,570" />
      <path d="M-50,170 C210,250 380,90 600,160 C820,230 930,80 1250,180" />
      <path d="M-50,130 C220,210 390,50 610,120 C830,190 940,40 1250,140" />
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
        <div className="falcon" ref={falcon} aria-hidden="true">
          {/* an emblem-style falcon flying left: solid grey shapes with carved gaps (each shape has a ground-coloured
              outline, so where shapes meet a thin gap appears). The head: a hooked beak growing out of the forehead
              (drawn at 70% about its root), an angry eye under a heavy brow, a mouth crescent under the cheek; the
              feet are tucked back under the tail, three hooked talons at the end */}
          <svg viewBox="10 0 340 190">
            <path className="f" d="M264,124 C288,118 312,112 338,107 L328,121 L344,127 L330,138 L340,152 C314,152 290,147 264,142 Z" />
            <path className="cut" d="M278,128 C296,126 314,124 332,124 C314,128 296,130 280,131 Z M278,137 C296,138 314,140 332,143 C314,142 296,141 280,140 Z" />
            <path className="f" d="M50,112 C62,92 92,84 122,96 C150,108 180,116 214,120 C236,123 254,124 270,124 L268,142 C246,148 220,154 196,156 C160,158 128,156 104,150 C84,145 68,138 58,128 C53,122 50,117 50,112 Z" />
            <path className="cut" d="M118,137 C146,148 176,152 206,150 C176,147 148,142 124,132 Z" />
            <path className="f" d="M180,148 C194,158 214,164 238,166 C226,172 206,174 192,172 C184,168 178,160 180,148 Z" />
            <path className="solid" d="M236,164 C243,164 248,168 248,175 C245,171 241,170 237,170 Z M232,170 C238,172 241,177 240,183 C237,178 234,176 230,175 Z M226,172 C230,176 231,181 229,186 C227,181 225,179 222,177 Z" />
            <g className="falcon-wing" style={{ transformOrigin: "160px 118px" }}>
              <path className="f" d="M116,102 C128,70 156,44 194,28 C230,14 266,8 306,5 Q284,16 270,23 Q280,26 286,30 Q268,34 254,41 Q262,45 268,51 Q252,54 240,59 Q246,64 250,71 Q238,73 228,79 Q232,85 234,92 Q226,94 220,99 C216,110 212,118 208,124 C178,118 150,110 116,102 Z" />
              <path className="cut" d="M198,40 C222,32 246,26 270,23 C246,30 222,38 204,46 Z M202,56 C220,50 238,45 254,41 C238,48 220,56 208,62 Z M205,72 C218,67 230,62 240,59 C230,66 218,72 210,78 Z M207,88 C214,85 222,82 228,79 C222,85 214,90 211,93 Z M134,92 C152,70 172,54 198,42 C176,58 158,76 144,96 Z" />
            </g>
            <path className="solid" transform="translate(70 115) scale(0.7) translate(-70 -115)" d="M70,100 C50,96 30,104 22,119 C16,130 18,142 26,150 C27,140 33,133 43,130 C52,128 61,129 70,132 C66,122 64,110 70,100 Z" />
            <path className="cut" d="M76,97 C88,90 104,87 120,89 C106,92 92,96 80,102 Z" />
            <path className="cut" d="M70,111 L108,99 C103,109 94,116 83,119 C77,119 72,116 70,111 Z" />
            <circle className="pupil" cx="80.5" cy="111.5" r="3.2" />
            <path className="cut" transform="translate(70 115) scale(0.7) translate(-70 -115)" d="M43,131 C58,128 72,131 86,136 C96,140 106,144 117,146 C103,146 90,143 79,139 C67,135 55,133 45,134 Z" />
          </svg>
        </div>
      </div>
      {step != null && <i hidden data-step={step} data-step-at={0} />}
    </div>
  );
}
