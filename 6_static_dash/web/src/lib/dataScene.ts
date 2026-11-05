/** Where every card, split header and group cluster sits in the three stages of the Data page's grouping scene:
 *  A as D1 shipped it, B grouped into copies, C split by whole group. Pure, so it can be tested without a browser. */
export type Split = "train" | "val" | "test";
export type Pt = [number, number];
export type Head = [x: number, y: number, w: number];
export interface SceneCard { gid: string; orig: Split; newSplit: Split }
export interface SceneLayout {
  S: number;          // card size in px
  tight: boolean;     // small cards: the tag stacks under the image
  phone: boolean;     // splits stack as rows instead of columns
  A: Pt[]; B: Pt[]; C: Pt[];
  HA: Head[]; HC: Head[];
  clusters: Record<string, [x: number, y: number, w: number, h: number]>;
}

export const SPLITS: Split[] = ["train", "val", "test"];
const idx = (s: Split) => SPLITS.indexOf(s);

/** `W`×`H` is the stage; `capH` is the height kept free for the caption at the bottom. */
export function layoutScene(cards: SceneCard[], W: number, H: number, capH: number): SceneLayout {
  return W < 640 ? phone(cards, W, H, capH) : wide(cards, W, H, capH);
}

const gids = (cards: SceneCard[]) => [...new Set(cards.map((c) => c.gid))];
const rank = (cards: SceneCard[], c: SceneCard) => cards.filter((d) => d.gid === c.gid).indexOf(c); // position within its group

function wide(cards: SceneCard[], W: number, H: number, capH: number): SceneLayout {
  const g = 12, G = 40;
  const S = Math.floor(Math.max(30, Math.min(145, (W - 2 * G - 5 * g) / 8, (H - capH - 150) / 2 - 28)));
  const tight = S < 110, step = S + g, rowH = S + (tight ? 44 : 28), blockH = 52 + 2 * rowH;
  const y0 = Math.max(70, (H - capH - blockH) / 2 + 20), cardY = y0 + 52;
  const colsA = [4, 2, 2], colsC = [3, 2, 3];
  // columns sized to their cards, centred as a block
  const xs = (cols: number[]) => {
    const w = cols.map((n) => n * S + (n - 1) * g);
    let x = (W - w.reduce((a, b) => a + b, 0) - 2 * G) / 2;
    return w.map((wi): [number, number] => { const r: [number, number] = [x, wi]; x += wi + G; return r; });
  };
  const XA = xs(colsA), XC = xs(colsC);
  const A: Pt[] = [], B: Pt[] = [], C: Pt[] = [];
  SPLITS.forEach((sp, ci) => cards.filter((c) => c.orig === sp).forEach((c, k) =>
    (A[cards.indexOf(c)] = [XA[ci][0] + (k % colsA[ci]) * step, cardY + Math.floor(k / colsA[ci]) * rowH])));
  const cw = 3 * S + 2 * g + 28, ch = 2 * rowH + 76, gapC = 80, cx1 = (W - 2 * cw - gapC) / 2;
  const clusters: SceneLayout["clusters"] = {};
  gids(cards).forEach((gid, n) => (clusters[gid] = [cx1 + n * (cw + gapC), cardY - 4, cw, ch]));
  cards.forEach((c, i) => {
    const k = rank(cards, c), [x, y] = clusters[c.gid];
    B[i] = [x + 14 + (k % 3) * step, y + 60 + Math.floor(k / 3) * rowH];
    C[i] = [XC[idx(c.newSplit)][0] + (k % 3) * step, cardY + Math.floor(k / 3) * rowH];
  });
  return { S, tight, phone: false, A, B, C, HA: XA.map(([x, w]) => [x, y0, w]), HC: XC.map(([x, w]) => [x, y0, w]), clusters };
}

// phones: the splits stack as rows, six cards to a line (validation and test share a row while both are small);
// each group's cluster is a single line
function phone(cards: SceneCard[], W: number, H: number, capH: number): SceneLayout {
  const g = 6, per = 6, headH = 40, gapR = 14, top = 56, emptyH = 56, avail = H - capH;
  const S = Math.floor(Math.max(30, Math.min(
    (W - (per - 1) * g - 24) / per,                                  // a cluster line, with its padding
    (avail - top - 2 * headH - gapR) / 3 - 50,                       // as shipped: train on two lines, val | test on one
    (avail - top - 3 * headH - 2 * gapR - emptyH) / 2 - 50,          // split by group: train, empty val, test
    (avail - top - 2 * 60 - 12) / 2 - 50,                            // two one-line clusters
  )));
  const step = S + g, rowH = S + 50, half = per / 2; // the tag may wrap to three short lines: 519 / was / train
  const rows = (splitOf: (c: SceneCard) => Split) => {
    const n = SPLITS.map((sp) => cards.filter((c) => splitOf(c) === sp).length);
    const heads: Head[] = [];
    let y = top;
    SPLITS.forEach((sp, ci) => {
      if (sp === "test" && n[1] && n[1] <= half && n[2] <= half) { heads[2] = [half * step, heads[1][1], W - half * step]; return; }
      const paired = sp === "val" && n[1] && n[1] <= half && n[2] <= half;
      heads[ci] = [0, y, paired ? half * step - g : W];
      y += headH + (n[ci] ? Math.ceil(n[ci] / per) * rowH : emptyH) + gapR;
    });
    return heads;
  };
  const place = (heads: Head[], splitOf: (c: SceneCard) => Split) => {
    const out: Pt[] = [];
    SPLITS.forEach((sp, ci) => cards.filter((c) => splitOf(c) === sp)
      .forEach((c, k) => (out[cards.indexOf(c)] = [heads[ci][0] + (k % per) * step, heads[ci][1] + headH + Math.floor(k / per) * rowH])));
    return out;
  };
  const HA = rows((c) => c.orig), HC = rows((c) => c.newSplit);
  const cw = per * S + (per - 1) * g + 24, ch = rowH + 60, cx = (W - cw) / 2;
  const clusters: SceneLayout["clusters"] = {};
  gids(cards).forEach((gid, n) => (clusters[gid] = [cx, top + n * (ch + 12), cw, ch]));
  const B = cards.map((c): Pt => { const k = rank(cards, c), [x, y] = clusters[c.gid]; return [x + 12 + k * step, y + 58]; });
  return { S, tight: true, phone: true, A: place(HA, (c) => c.orig), B, C: place(HC, (c) => c.newSplit), HA, HC, clusters };
}
