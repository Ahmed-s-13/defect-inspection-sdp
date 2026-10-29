import type React from "react";
import { className, int, pct } from "../../lib/format";

/** The charts behind the pipeline scene's clickable stages (scripts/pipeline_scene.py). Plain SVG, like the other
 *  charts; the current crop is the ringed mark. */

const W = 360;

/** The calibration crops' distance to their 5 nearest training crops, with where the kNN part alone stops a crop. */
export function KnnHist({ knn, at, alpha }: { knn: { edges: number[]; counts: number[]; n: number; cut: Record<string, number> }; at: number; alpha: number }) {
  const H = 150, top = 14, base = 118, lo = knn.edges[0], hi = knn.edges[knn.edges.length - 1];
  const x = (v: number) => 8 + ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (W - 16);
  const max = Math.max(...knn.counts), cut = knn.cut[alpha.toFixed(2)];
  return (
    <figure className="pl-chart">
      <figcaption><b>How far from the training crops?</b> Distance from each of the {int(knn.n)} calibration crops to its 5 nearest training crops. Past the line, this part alone stops a crop.</figcaption>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Histogram of calibration distances with the stop line and this crop">
        {knn.counts.map((n, i) => {
          const h = (n / max) * (base - top);
          return <rect key={i} className="bar" x={x(knn.edges[i]) + 0.5} y={base - h} width={Math.max(x(knn.edges[i + 1]) - x(knn.edges[i]) - 1, 1)} height={h} />;
        })}
        <line className="ref" x1={x(cut)} x2={x(cut)} y1={top - 6} y2={base} />
        <text className="tick" x={x(cut) + 4} y={top} textAnchor={cut > hi ? "end" : "start"}>{cut > hi ? "stop line further right →" : "stops here →"}</text>
        <line className="pl-mark" x1={x(at)} x2={x(at)} y1={top + 4} y2={base + 4} />
        <circle className="pl-ring" cx={x(at)} cy={base + 4} r="5" />
        {x(at) - x(lo) > 70 && <text className="tick" x={x(lo)} y={H - 4}>closer</text>}
        {x(hi) - x(at) > 70 && <text className="tick" x={x(hi)} y={H - 4} textAnchor="end">further</text>}
        <text className="tick" x={Math.min(Math.max(x(at), 30), W - 30)} y={H - 4} textAnchor="middle">this crop</text>
      </svg>
    </figure>
  );
}

/** Aircraft-skin crops (the model's training crops) against the off-target photos the aircraft-or-not check was
 *  fitted on. Across: its score (log-odds of "not aircraft skin"); down: the features' main other direction. */
export function AonScatter({ aon, at, alpha }: { aon: { skin: [number, number][]; off: [number, number][]; cut: Record<string, number> }; at: [number, number]; alpha: number }) {
  const H = 220, pad = 10, all = [...aon.skin, ...aon.off, at];
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const x = (v: number) => pad + ((v - x0) / (x1 - x0)) * (W - 2 * pad), y = (v: number) => pad + ((v - y0) / (y1 - y0)) * (H - 2 * pad - 14);
  const cut = aon.cut[alpha.toFixed(2)];
  return (
    <figure className="pl-chart">
      <figcaption><b>Does it look like aircraft skin?</b> A logistic check on the model's features, fitted on its training crops and on photos of people, tools, vehicles and scenes. Right of the line, this part alone stops a crop.</figcaption>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Scatter of aircraft-skin and off-target crops with the stop line and this crop">
        {aon.skin.map((p, i) => <circle key={`s${i}`} className="pl-dot-skin" cx={x(p[0])} cy={y(p[1])} r="2.2" />)}
        {aon.off.map((p, i) => <circle key={`o${i}`} className="pl-dot-off" cx={x(p[0])} cy={y(p[1])} r="2.2" />)}
        <line className="ref" x1={x(cut)} x2={x(cut)} y1={pad} y2={H - 22} />
        <circle className="pl-ring" cx={x(at[0])} cy={y(at[1])} r="6" />
        <text className="tick" x={pad} y={H - 4}>aircraft skin</text>
        <text className="tick" x={W - pad} y={H - 4} textAnchor="end">not aircraft skin</text>
      </svg>
      <p className="chart-key"><i className="pl-key pl-dot-skin" /> Training crops <i className="pl-key pl-dot-off" /> Off-target photos <i className="pl-key pl-key-ring" /> This crop</p>
    </figure>
  );
}

/** How often the served set holds the true class, class by class, against the target. */
export function ClassCoverage({ cov, target }: { cov: Record<string, number>; target: number }) {
  const rows = Object.entries(cov).sort((a, b) => a[1] - b[1]);
  return (
    <figure className="pl-chart">
      <figcaption><b>The {pct(target, 0)} is an average.</b> How often the answer holds the true class, for each class, on the test crops.</figcaption>
      <ol className="pl-cov" style={{ "--target": target } as React.CSSProperties}>
        {rows.map(([c, v]) => (
          <li key={c}><span>{className(c)}</span><span className="pl-cov-track"><i style={{ transform: `scaleX(${v})` }} /><b /></span><span className="fig">{pct(v, 0)}</span></li>
        ))}
      </ol>
      <p className="chart-key"><i className="pl-key pl-key-target" /> Target {pct(target, 0)}</p>
    </figure>
  );
}
