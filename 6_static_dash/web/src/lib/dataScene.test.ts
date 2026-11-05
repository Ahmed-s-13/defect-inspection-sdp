import { describe, expect, it } from "vitest";
import { layoutScene, SPLITS, type Pt, type SceneCard, type SceneLayout } from "./dataScene";

// the two real groups on the Data page: 148 goes to train, 519 to test
const orig148 = ["train", "train", "train", "train", "val", "test"] as const;
const orig519 = ["train", "train", "train", "train", "val", "test"] as const;
const CARDS: SceneCard[] = [
  ...orig148.map((o) => ({ gid: "148", orig: o, newSplit: "train" as const })),
  ...orig519.map((o) => ({ gid: "519", orig: o, newSplit: "test" as const })),
];

const overlaps = (pts: Pt[], w: number, h: number) =>
  pts.some((a, i) => pts.some((b, j) => i < j && Math.abs(a[0] - b[0]) < w && Math.abs(a[1] - b[1]) < h));
const tagH = (L: SceneLayout) => (L.phone ? 48 : L.tight ? 36 : 20);

describe.each([[1440, 844, 130], [814, 704, 130], [343, 712, 166], [343, 600, 166]])("layoutScene at %ix%i", (W, H, capH) => {
  const L = layoutScene(CARDS, W, H, capH);

  it("keeps every card on the stage and clear of the caption in every stage", () => {
    for (const pts of [L.A, L.B, L.C]) {
      expect(pts).toHaveLength(CARDS.length);
      for (const [x, y] of pts) {
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x + L.S).toBeLessThanOrEqual(W);
        expect(y + L.S + tagH(L)).toBeLessThanOrEqual(H - capH);
      }
    }
  });

  it("never stacks one card on another", () => {
    for (const pts of [L.A, L.B, L.C]) expect(overlaps(pts, L.S, L.S)).toBe(false);
  });

  it("puts each card under its split's header: shipped split first, group's split after", () => {
    CARDS.forEach((c, i) => {
      for (const [pts, heads, sp] of [[L.A, L.HA, c.orig], [L.C, L.HC, c.newSplit]] as const) {
        const [hx, hy, hw] = heads[SPLITS.indexOf(sp)];
        expect(pts[i][0]).toBeGreaterThanOrEqual(hx - 0.5);
        expect(pts[i][0] + L.S).toBeLessThanOrEqual(hx + hw + 0.5);
        expect(pts[i][1]).toBeGreaterThan(hy);
      }
    });
  });

  it("holds each group's cards inside its cluster", () => {
    CARDS.forEach((c, i) => {
      const [x, y, w, h] = L.clusters[c.gid], [px, py] = L.B[i];
      expect(px).toBeGreaterThan(x);
      expect(py).toBeGreaterThan(y);
      expect(px + L.S).toBeLessThan(x + w);
      expect(py + L.S).toBeLessThan(y + h);
    });
  });
});

it("stacks the splits as rows on a phone", () => {
  expect(layoutScene(CARDS, 343, 700, 166).phone).toBe(true);
  expect(layoutScene(CARDS, 814, 704, 130).phone).toBe(false);
});
