import type React from "react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { CleanCropSection, Gc7Section } from "../components/CleanCropSection";
import { DatasetSection } from "../components/DatasetSection";
import { HealerSection } from "../components/HealerSection";
import crop from "../data/crop.json";
import meta from "../data/grouping.json";
import { layoutScene, type SceneCard, type SceneLayout, type Split } from "../lib/dataScene";
import { useReveal } from "../lib/useReveal";
import { usePresent } from "../state/present";
import "../styles/data.css";
import { watchNear } from "../lib/near";

const IMG = "data-grouping/";
const vars = (v: Record<string, string | number>) => v as React.CSSProperties;

type Card = SceneCard & { file: string };
const GROUPS = ["148", "519"] as const;
const CARDS: Card[] = GROUPS.flatMap((gid) => meta.groups[gid].items.map((it) => ({ gid, file: it.file, orig: it.orig as Split, newSplit: meta.groups[gid].new as Split })));

// presentation steps, in the order we did the work. What was wrong with D1; part 1, a leak-free split: the scene's three
// stages and the four steps of how we did it; part 2, the no-defect class: the healer (before/after, how it works),
// the clean cropper (its five stages, how it picks) and d1_gc7; part 3, from photo to datapoint: the crop's four stages
const SCENE_STEPS = [1, 2, 3], HOW_FIRST = 4, HEAL_FIRST = 8, CLEAN_FIRST = 10, GC7_STEP = 16, CROP_STEPS = [17, 18, 19, 20], DS_FIRST = 21, STEP_COUNT = 23;

export function DataView() {
  const present = usePresent();
  const dimAt = (...steps: number[]) => present.on && !steps.includes(present.step);
  return (
    <div className="data">
      <header className="enter" style={vars({ "--i": 0 })}>
        <h1 className="view-title">Data</h1>
        <p className="view-sub data-lede">D1 ships 14,760 photos in six defect classes. Most are augmented copies of a few hundred shots, its own split put copies of the same photo in training and in test, and none of its photos shows undamaged skin. This page follows the steps we took to fix both, and then how a photo becomes the model&rsquo;s input.</p>
      </header>
      <WhyD1 dim={dimAt(0)} />
      <h2 className="data-part"><small>Part 1</small>A leak-free split: d1_grouped</h2>
      <p className="how-sub">First we found every copy of every photo and kept each scene on one side of the split.</p>
      <GroupingScene dim={dimAt(...SCENE_STEPS)} />
      <HowWeDidIt first={HOW_FIRST} dimAt={dimAt} />
      <h2 className="data-part"><small>Part 2</small>A 7th class: no defect</h2>
      <p className="how-sub">With only defect classes, a model must name a defect even on clean paint. We made intact-surface crops from D1&rsquo;s own photos in two ways, both checked by hand, and added them as a 7th class.</p>
      <HealerSection first={HEAL_FIRST} dimAt={dimAt} />
      <CleanCropSection first={CLEAN_FIRST} dimAt={dimAt} />
      <Gc7Section step={GC7_STEP} dim={dimAt(GC7_STEP)} />
      <h2 className="data-part"><small>Part 3</small>From photo to datapoint</h2>
      <CropScene dim={dimAt(...CROP_STEPS)} />
      <h2 className="data-part"><small>Part 4</small>Every dataset we used</h2>
      <p className="how-sub">The photos behind the model, its calibration and its tests, with their sizes, where each was used, and real samples.</p>
      <DatasetSection first={DS_FIRST} dimAt={dimAt} />
      {present.on && <p className="step-hint">Section {present.step + 1} of {STEP_COUNT} · ← → to move</p>}
    </div>
  );
}

function WhyD1({ dim }: { dim: boolean }) {
  const rv = useReveal<HTMLElement>(0.85);
  return (
    <section className="why enter present-dim" style={vars({ "--i": 1 })} data-step={0} data-dim={dim} ref={rv.ref} data-shown={rv.shown}>
      <h2 className="data-h2 rv" style={vars({ "--i": 0 })}>What was wrong with D1</h2>
      <div className="facts3">
        <div className="fact rv" style={vars({ "--i": 1 })}>
          <span className="fact-big">861 <small>distinct scenes</small></span>
          <p>behind 14,760 photos. The rest are flipped, rotated, cropped or recoloured copies, or near-identical re-shoots.</p>
          <div className="meter"><i className="ink" style={vars({ "--w": 0.0583, "--k": 0 })} /></div>
          <div className="meter-key"><span>scenes 6%</span><span>copies 94%</span></div>
        </div>
        <div className="fact rv" style={vars({ "--i": 2 })}>
          <span className="fact-big">87.6% <small>of test photos</small></span>
          <p>had a copy in the training set, so the test mostly asked the model about pictures it had already seen.</p>
          <div className="meter"><i className="red" style={vars({ "--w": 0.876, "--k": 1 })} /></div>
          <div className="meter-key"><span>copy in training</span><span>1,293 of 1,476</span></div>
        </div>
        <div className="fact rv" style={vars({ "--i": 3 })}>
          <span className="fact-big">12 <small>points too high</small></span>
          <p>The same model scored 90.1% on D1's own split and 78.1% once no test photo had a copy in training.</p>
          <div className="two-bars">
            <div className="row"><span>D1's split</span><div className="meter"><i className="grey" style={vars({ "--w": 0.901, "--k": 2 })} /></div><b>90.1%</b></div>
            <div className="row"><span>Leak-free</span><div className="meter"><i className="ink" style={vars({ "--w": 0.781, "--k": 3 })} /></div><b>78.1%</b></div>
          </div>
        </div>
      </div>
      <div className="fact-wide rv" style={vars({ "--i": 4 })}>
        <span className="fact-big">0 <small>photos of intact skin</small></span>
        <p>All six of D1&rsquo;s classes are defects, so a model trained on it names a defect on every square it sees, even clean paint. Inspection needs &ldquo;nothing wrong here&rdquo; as an answer.</p>
      </div>
    </section>
  );
}

const CAPTIONS = [
  ["As shipped, the same photo sits in training and in test.", "Flipped, rotated and recoloured copies were scattered across D1's split: 87.6% of its test photos have a copy in training."],
  ["We find every copy and group them.", "Each group is one scene: a photo and all its copies. All of D1 becomes 861 groups."],
  ["Whole groups go to one split. Nothing crosses.", "Group 148 is all training; group 519 is all test. The model is tested on photos it has never seen in any form."],
];
const STAGES = ["As shipped", "Copies found", "Split by group"];
const HEADS: [string, string][] = [["Train", "70%"], ["Validation", "15%"], ["Test", "15%"]];
const ease3 = (t: number) => 1 - Math.pow(1 - t, 3);
const clamp = (v: number) => Math.max(0, Math.min(1, v));

/** Where each stage reads best, as a fraction of the scroll through the scene; presentation mode stops here. */
const STAGE_AT = [0.04, 0.48, 0.96];

/** A tall run with a pinned stage: scrolling moves the 12 photos from D1's split, into their two groups, into the new split. */
function GroupingScene({ dim }: { dim: boolean }) {
  const run = useRef<HTMLElement>(null), stage = useRef<HTMLDivElement>(null), caption = useRef<HTMLDivElement>(null);
  const cardEls = useRef<(HTMLDivElement | null)[]>([]), headEls = useRef<(HTMLDivElement | null)[]>([]);
  const clusterEls = useRef<(HTMLDivElement | null)[]>([]), edgeEls = useRef<(SVGLineElement | null)[]>([]);
  const [phase, setPhase] = useState(0);
  const [layout, setLayout] = useState<Pick<SceneLayout, "S" | "tight" | "phone"> | null>(null);
  const remeasure = useRef<() => void>(() => undefined);
  const { on: presenting } = usePresent();
  // presentation mode enlarges the caption; measure again once its type has changed (after the provider's effect)
  useEffect(() => { const r = requestAnimationFrame(() => remeasure.current()); return () => cancelAnimationFrame(r); }, [presenting]);

  useLayoutEffect(() => {
    const runEl = run.current!, st = stage.current!, cap = caption.current!;
    let L: SceneLayout | null = null, raf = 0, lastPhase = -1;
    const measure = () => {
      const bar = document.querySelector<HTMLElement>(".topbar")?.offsetHeight ?? 56;
      st.style.setProperty("--stick", `${bar}px`); // pinned under the top bar
      st.dataset.phone = String(st.clientWidth < 640); // set here, not in JSX: the caption below is measured in its phone size
      const tallest = Math.max(...Array.from(cap.children, (p) => (p as HTMLElement).offsetHeight));
      cap.style.height = `${tallest}px`;
      L = layoutScene(CARDS, st.clientWidth, st.clientHeight, st.clientHeight - cap.offsetTop + 8);
      st.style.setProperty("--s", `${L.S}px`);
      setLayout({ S: L.S, tight: L.tight, phone: L.phone });
    };
    const frame = () => {
      raf = 0;
      if (!L) measure();
      const lay = L!, span = runEl.offsetHeight - st.offsetHeight;
      const p = clamp((st.getBoundingClientRect().top - runEl.getBoundingClientRect().top) / span);
      const ab = ease3(clamp((p - 0.18) / 0.2)), bc = ease3(clamp((p - 0.58) / 0.2));
      const centres = CARDS.map((_, k) => {
        const d = (k % 6) * 0.025; // a small ripple across each group
        const t1 = ease3(clamp((p - 0.18 - d) / 0.2)), t2 = ease3(clamp((p - 0.58 - d) / 0.2));
        const a = lay.A[k], b = lay.B[k], c = lay.C[k];
        const x = a[0] + (b[0] - a[0]) * t1 + (c[0] - b[0]) * t2, y = a[1] + (b[1] - a[1]) * t1 + (c[1] - b[1]) * t2;
        const el = cardEls.current[k];
        if (el) el.style.transform = `translate(${x}px, ${y}px)`;
        return [x + lay.S / 2, y + lay.S / 2];
      });
      // split headers: shipped place -> fade while grouped -> new place
      const headOp = String(Math.max(lay.phone ? 0 : 0.1, Math.min(1, 1 - ab + bc))); // on phones the clusters cover the headers
      headEls.current.forEach((h, i) => {
        if (!h) return;
        const [xa, ya, wa] = lay.HA[i], [xc, yc, wc] = lay.HC[i];
        h.style.transform = `translate(${xa + (xc - xa) * bc}px, ${ya + (yc - ya) * bc}px)`;
        h.style.width = `${wa + (wc - wa) * bc}px`;
        h.style.opacity = headOp;
      });
      st.style.setProperty("--after", String(bc));
      GROUPS.forEach((gid, n) => {
        const el = clusterEls.current[n], [x, y, w, h] = lay.clusters[gid];
        if (el) Object.assign(el.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px`, opacity: String(Math.max(0, ab - bc * 1.2)) });
      });
      // faint lines joining each copy to the next while grouped
      const edgeOp = String(Math.max(0, ab - bc) * 0.45);
      let e = 0;
      GROUPS.forEach((gid) => {
        const ks = CARDS.flatMap((c, k) => (c.gid === gid ? [k] : []));
        ks.slice(1).forEach((k, j) => {
          const l = edgeEls.current[e++];
          if (!l) return;
          const [x1, y1] = centres[ks[j]], [x2, y2] = centres[k];
          l.setAttribute("x1", String(x1)); l.setAttribute("y1", String(y1)); l.setAttribute("x2", String(x2)); l.setAttribute("y2", String(y2));
          l.style.opacity = edgeOp;
        });
      });
      const ph = p < 0.32 ? 0 : p < 0.7 ? 1 : 2;
      if (ph !== lastPhase) { lastPhase = ph; setPhase(ph); }
    };
    // static dash: nothing to do while the scene is far off screen; one last frame as it leaves
    const near = watchNear(runEl, () => { if (!raf) raf = requestAnimationFrame(frame); });
    const onScroll = () => { if (near.near && !raf) raf = requestAnimationFrame(frame); };
    const onResize = () => { L = null; onScroll(); };
    remeasure.current = onResize;
    frame();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onResize);
    void document.fonts?.ready.then(onResize); // caption height depends on the loaded fonts
    return () => { near.stop(); cancelAnimationFrame(raf); window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onResize); };
  }, []);

  const edges = GROUPS.flatMap((gid) => CARDS.filter((c) => c.gid === gid).slice(1));
  return (
    <section className="scene-run present-dim" data-dim={dim} ref={run} aria-label="How the same photos move from D1's split into groups and a new split">
      {STAGE_AT.map((at, i) => <i key={at} className="stage-mark" data-step={SCENE_STEPS[i]} data-step-at={at} aria-hidden="true" />)}
      <div className={`scene${phase === 2 ? " after-split" : ""}${layout?.tight ? " tight" : ""}`} ref={stage}>
        {HEADS.map(([name, share], i) => (
          <div key={name} className="colhead" ref={(el) => { headEls.current[i] = el; }}>
            <h3>{name} <span className="share">{share}</span></h3>
            {i === 1 && <div className="empty-note">No copies of these two photos land here.</div>}
          </div>
        ))}
        {GROUPS.map((gid, n) => (
          <div key={gid} className="cluster" ref={(el) => { clusterEls.current[n] = el; }}>
            <b>Group {gid}</b><span>6 copies of one photo · {meta.groups[gid].cls}</span>
          </div>
        ))}
        <svg className="links" aria-hidden="true">
          {edges.map((c, k) => <line key={`${c.gid}-${c.file}`} className="edge" ref={(el) => { edgeEls.current[k] = el; }} />)}
        </svg>
        {CARDS.map((c, k) => (
          <div key={c.file} className="card" ref={(el) => { cardEls.current[k] = el; }}>
            <img src={IMG + c.file} alt="" />
            <div className="tag"><b>{c.gid}</b><span className="o">{c.orig}</span></div>
          </div>
        ))}
        <div className="caption" ref={caption} aria-live="polite">
          {CAPTIONS.map(([head, sub], i) => <p key={head} className={phase === i ? "on" : undefined}><strong>{head}</strong><span>{sub}</span></p>)}
        </div>
        <div className="progress" aria-hidden="true">{STAGES.map((s, i) => <span key={s} className={phase === i ? "on" : undefined}>{s}</span>)}</div>
      </div>
    </section>
  );
}

function HowWeDidIt({ first, dimAt }: { first: number; dimAt(step: number): boolean }) {
  return (
    <section className="how">
      <h2 className="data-h2">How we did it</h2>
      <p className="how-sub">Checking every pair of D1's 14,760 photos point by point would mean 109 million comparisons. Each step narrows the work, and its result is the next step's input.</p>
      <Fingerprint n={first} dim={dimAt(first)} />
      <ConfirmCopies n={first + 1} dim={dimAt(first + 1)} />
      <JoinGroups n={first + 2} dim={dimAt(first + 2)} />
      <SplitGroups n={first + 3} dim={dimAt(first + 3)} />
    </section>
  );
}

/** One step: what it does on the left, its figure on the right; `from → to` names what it takes in and hands on. */
function Step({ n, k, dim, title, text, from, to, children }: {
  n: number; k: number; dim: boolean; title: string; text: React.ReactNode; from: string; to: string; children: React.ReactNode;
}) {
  const rv = useReveal<HTMLDivElement>(0.85);
  return (
    <div className="step present-dim" data-step={n} data-dim={dim} ref={rv.ref} data-shown={rv.shown}>
      <div>
        <h3 className="rv" style={vars({ "--i": 0 })}><span className="step-n">{k}</span>{title}</h3>
        <p className="rv" style={vars({ "--i": 1 })}>{text}</p>
        <p className="step-io rv" style={vars({ "--i": 2 })}><span>{from}</span><span className="arrow" aria-label="becomes">→</span><b>{to}</b></p>
      </div>
      <div className="rv" style={vars({ "--i": 2 })}>{children}</div>
    </div>
  );
}

function Fingerprint({ n, dim }: { n: number; dim: boolean }) {
  const first = meta.groups["148"].items[0].file;
  return (
    <Step n={n} k={1} dim={dim} title="Fingerprint every photo to shortlist look-alikes" from="109 million possible pairs" to="260,113 shortlisted pairs"
      text={<>Each photo gets two short fingerprints, taken in all 4 rotations and their mirror images so a turned copy still matches: a 64-bit perceptual hash of its light and dark areas, and a <b>ResNet-50 embedding</b>, 2,048 numbers from a network pretrained on ImageNet that describe what the photo shows. Each photo&rsquo;s closest fingerprints become its shortlist of possible copies.</>}>
      <div className="views8">
        {Array.from({ length: 8 }, (_, k) => (
          <div key={k} className="v"><img src={IMG + first} alt="" style={{ transform: `${k >= 4 ? "scaleX(-1) " : ""}rotate(${-90 * (k % 4)}deg)` }} /></div>
        ))}
      </div>
      <div className="prints">
        <div>
          <div className="hashgrid">{meta.phash_bits.map((b, k) => <i key={k} className={b ? "one" : undefined} style={vars({ "--k": k })} />)}</div>
          <span className="fig">perceptual hash · 64 bits</span>
        </div>
        <div>
          {/* bar height is the square root of each value, so the many small values still show */}
          <div className="emb">{meta.emb64.map((v, k) => <i key={k} style={vars({ "--h": Math.max(Math.sqrt(v), 0.04), "--k": k })} />)}</div>
          <span className="fig">ResNet-50 embedding · first 64 of 2,048</span>
        </div>
      </div>
    </Step>
  );
}

function ConfirmCopies({ n, dim }: { n: number; dim: boolean }) {
  const sf = meta.sift, gap = 36, sA = 260 / Math.max(...sf.aSize), sB = 260 / Math.max(...sf.bSize);
  const wA = sf.aSize[0] * sA, hA = sf.aSize[1] * sA, wB = sf.bSize[0] * sB, hB = sf.bSize[1] * sB;
  return (
    <Step n={n} k={2} dim={dim} title="Confirm each shortlisted pair point by point" from="260,113 shortlisted pairs" to="116,879 confirmed copies"
      text={<>Close fingerprints only mean two photos look alike. For each shortlisted pair, SIFT matches small features between the two and checks they agree on one rotation and shift. Real copies pass, even cropped or recoloured ones; different damage that only looks similar does not. This pair shares {sf.inliers_total.toLocaleString("en-US")} matching points.</>}>
      <div className="sift">
        <svg viewBox={`0 0 ${wA + gap + wB} ${Math.max(hA, hB)}`} aria-hidden="true">
          <image href={IMG + sf.a} width={wA} height={hA} />
          <image href={IMG + sf.b} x={wA + gap} width={wB} height={hB} />
          {sf.lines.map(([[x1, y1], [x2, y2]], k) => {
            const X1 = x1 * sA, Y1 = y1 * sA, X2 = wA + gap + x2 * sB, Y2 = y2 * sB;
            return (
              <g key={k} style={vars({ "--k": k })}>
                <line className="m" pathLength={1} x1={X1} y1={Y1} x2={X2} y2={Y2} />
                <circle className="mdot" cx={X1} cy={Y1} r={2.6} /><circle className="mdot" cx={X2} cy={Y2} r={2.6} />
              </g>
            );
          })}
        </svg>
        <div className="sift-tags"><span>a {sf.aOrig} photo</span><span>a {sf.bOrig} photo, turned {sf.rotation_deg}°</span></div>
      </div>
    </Step>
  );
}

const PTS = [[60, 105], [140, 50], [150, 160], [235, 95], [320, 45], [370, 150]];
const LINKS = [[0, 1], [1, 3], [0, 2], [2, 3], [3, 4], [4, 5], [3, 5]];

function JoinGroups({ n, dim }: { n: number; dim: boolean }) {
  const items = meta.groups["148"].items;
  return (
    <Step n={n} k={3} dim={dim} title="Join confirmed copies into groups" from="116,879 confirmed copies" to="861 groups"
      text="Each confirmed pair is a link, and linked photos form one group: if A copies B and B copies C, all three are one group, even when A and C look different. Each group is one scene.">
      <div className="graph">
        <svg viewBox="0 0 440 210" aria-hidden="true">
          <path className="hull" d="M28,105 C30,30 110,14 160,18 C250,8 360,4 400,40 C430,80 425,190 365,195 C300,200 240,190 150,196 C80,200 26,170 28,105 Z" />
          {LINKS.map(([a, b], k) => <line key={k} className="e" pathLength={1} style={vars({ "--k": k })} x1={PTS[a][0]} y1={PTS[a][1]} x2={PTS[b][0]} y2={PTS[b][1]} />)}
          {PTS.map(([x, y], k) => (
            <g key={k}>
              <clipPath id={`dg-node-${k}`}><circle cx={x} cy={y} r={24} /></clipPath>
              <image href={IMG + items[k].file} x={x - 24} y={y - 24} width={48} height={48} preserveAspectRatio="xMidYMid slice" clipPath={`url(#dg-node-${k})`} />
              <circle className="node-ring" cx={x} cy={y} r={24} />
            </g>
          ))}
        </svg>
      </div>
    </Step>
  );
}

function SplitGroups({ n, dim }: { n: number; dim: boolean }) {
  const bars: [string, number, number][] = [["Train", 0.7, 502], ["Validation", 0.15, 184], ["Test", 0.15, 175]];
  return (
    <Step n={n} k={4} dim={dim} title="Split whole groups, then check again" from="861 groups" to="3 splits, no copy shared"
      text="Whole groups are dealt 70 / 15 / 15 for every defect class, so a group never spans two splits. Then every validation and test photo is compared with every photo in the other splits; any copy found is merged into its group and the split redone, until none are found.">
      <div className="split-bars">
        {bars.map(([name, w, n], k) => (
          <div key={name} className="split-bar"><span>{name}</span><span className="track"><span className="fill" style={vars({ "--w": w, "--k": k })} /></span><span className="n">{n} groups</span></div>
        ))}
      </div>
      <div className="rounds"><span>Copies found across splits, per round:</span><span className="r">117</span><span className="arrow">→</span><span className="r">1</span><span className="arrow">→</span><span className="r zero">0</span></div>
    </Step>
  );
}

const [bx1, by1, bx2, by2] = crop.box, [cl, ct, cs] = crop.crop;
const bw = Math.round(bx2 - bx1), bh = Math.round(by2 - by1);
const CROP_CAPTIONS: [string, React.ReactNode][] = [
  ["The full photo", <>{crop.w} × {crop.h} px from the test split. Most of it is grass, trees and panels away from the damage.</>],
  ["The label marks the defect", <>Every D1 photo comes with a box around each defect. This crack&rsquo;s box is {bw} × {bh} px.</>],
  ["Square it, with 25% to spare", <>We take the box&rsquo;s longer side and add 25% on every side: {bw} × 1.5 &asymp; {cs} px. The square stays centred on the defect; near the photo&rsquo;s edge it shifts inward instead of padding.</>],
  ["Resize to 224 × 224", <>The crop is resized to 224 × 224 px, the network&rsquo;s input size (this one is already {cs}). One crop per box, so a photo with three defects gives three crops. In Inspect there is no label: the square sits at the centre of each sample photo.</>],
];
const CROP_AT = [0.04, 0.34, 0.62, 0.96];

/** A pinned stage: scrolling marks the defect on a full photo, squares the box up, then zooms in until the frame holds exactly the model's crop. */
function CropScene({ dim }: { dim: boolean }) {
  const run = useRef<HTMLDivElement>(null), stage = useRef<HTMLDivElement>(null), zoom = useRef<HTMLDivElement>(null);
  const rect = useRef<SVGRectElement>(null), shade = useRef<SVGPathElement>(null), real = useRef<HTMLImageElement>(null);
  const [phase, setPhase] = useState(0);

  useLayoutEffect(() => {
    const runEl = run.current!, st = stage.current!;
    let raf = 0, lastPhase = -1;
    const frame = () => {
      raf = 0;
      const span = runEl.offsetHeight - st.offsetHeight;
      const p = clamp((st.getBoundingClientRect().top - runEl.getBoundingClientRect().top) / span);
      const t1 = ease3(clamp((p - 0.12) / 0.14)), t2 = ease3(clamp((p - 0.4) / 0.16)), t3 = ease3(clamp((p - 0.68) / 0.18));
      // the label's box grows into the square crop
      const x = bx1 + (cl - bx1) * t2, y = by1 + (ct - by1) * t2;
      const r = rect.current!;
      r.setAttribute("x", String(x)); r.setAttribute("y", String(y));
      r.setAttribute("width", String(bw + (cs - bw) * t2)); r.setAttribute("height", String(bh + (cs - bh) * t2));
      r.style.opacity = String(t1 * (1 - t3));
      r.style.stroke = t2 > 0.5 ? "var(--control)" : "var(--detected)";
      shade.current!.style.opacity = String(t2 * 0.55);
      // zoom: the visible window shrinks from the whole photo to the crop
      const wl = cl * t3, wt = ct * t3, ws = crop.w + (cs - crop.w) * t3, k = crop.w / ws;
      zoom.current!.style.transform = `scale(${k}) translate(${(-wl / crop.w) * 100}%, ${(-wt / crop.h) * 100}%)`;
      real.current!.style.opacity = String(clamp((p - 0.86) / 0.06)); // the saved crop itself, once the frame holds it
      const ph = p < 0.27 ? 0 : p < 0.52 ? 1 : p < 0.8 ? 2 : 3;
      if (ph !== lastPhase) { lastPhase = ph; setPhase(ph); }
    };
    // static dash: nothing to do while the scene is far off screen; one last frame as it leaves
    const near = watchNear(runEl, () => { if (!raf) raf = requestAnimationFrame(frame); });
    const onScroll = () => { if (near.near && !raf) raf = requestAnimationFrame(frame); };
    frame();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { near.stop(); cancelAnimationFrame(raf); window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, []);

  const hole = `M0 0H${crop.w}V${crop.h}H0Z M${cl} ${ct}v${cs}h${cs}v${-cs}Z`;
  return (
    <section className="crop-sec">
      <h2 className="data-h2">Cropping each defect</h2>
      <p className="how-sub crop-lede">Every datapoint in d1_gc7 is a 224 × 224 square. For a defect we cut a square around its label, as below. A healed crop is the same square cut from the healed photo, and a clean crop is the clean cropper&rsquo;s square, resized the same way.</p>
      <div className="scene-run crop-run present-dim" data-dim={dim} ref={run} aria-label="How a full photo becomes the model's 224 by 224 crop">
        {CROP_AT.map((at, i) => <i key={at} className="stage-mark" data-step={CROP_STEPS[i]} data-step-at={at} aria-hidden="true" />)}
        <div className="scene crop-stage" ref={stage}>
          <div className="crop-frame">
            <div className="crop-zoom" ref={zoom}>
              <img src={IMG + "crop_full.jpg"} alt={`A D1 test photo of a ${crop.cls} on a wing panel`} />
              <svg viewBox={`0 0 ${crop.w} ${crop.h}`} aria-hidden="true">
                <path ref={shade} className="crop-shade" d={hole} fillRule="evenodd" />
                <rect ref={rect} className="crop-rect" vectorEffect="non-scaling-stroke" />
              </svg>
            </div>
            <img ref={real} className="crop-real" src={IMG + "crop_224.jpg"} alt="" />
          </div>
          <ol className="crop-steps" aria-live="polite">
            {CROP_CAPTIONS.map(([head, text], i) => (
              <li key={head} className={phase === i ? "on" : undefined}><strong><span className="step-n">{i + 1}</span>{head}</strong><span><span>{text}</span></span></li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}
