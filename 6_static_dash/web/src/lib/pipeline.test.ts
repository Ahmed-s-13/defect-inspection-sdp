import { describe, expect, it } from "vitest";
import pipe from "../data/pipeline.json";

// scripts/pipeline_scene.py's output: the scene's five crops and the alpha grid behind "Try other alphas"
const key = (o: number, c: number) => `${o.toFixed(2)}|${c.toFixed(2)}`;
const served = key(pipe.served.ood, pipe.served.cp);
const grid = pipe.grid as Record<string, { coverage: number; set_size: number; reaim: number; mix: Record<string, number> }>;

describe("pipeline scene data", () => {
  it("has every alpha pair, the served one included", () => {
    for (const o of pipe.ood_alphas) for (const c of pipe.cp_alphas) expect(grid[key(o, c)]).toBeDefined();
    expect(grid[served]).toBeDefined();
  });
  it("has one crop per visor outcome, each landing in its own bucket as served", () => {
    expect(pipe.crops.map((c) => c.outcome).sort()).toEqual(["ambiguous", "nothing", "possible", "reaim", "show"]);
    for (const c of pipe.crops) expect((c.outcomes as Record<string, string>)[served]).toBe(c.outcome);
  });
  it("is architecture B as served: OOD 0.05, CP 0.10: coverage 91.0%, 1.54 classes, 1.5% re-aimed", () => {
    expect(pipe.served).toEqual({ ood: 0.05, cp: 0.10 });
    expect(grid[served].coverage).toBeCloseTo(0.9102, 3);
    expect(grid[served].set_size).toBeCloseTo(1.5363, 3);
    expect(grid[served].reaim).toBeCloseTo(0.0149, 3);
  });
  it("adds up, and a larger conformal alpha never raises coverage", () => {
    for (const g of Object.values(grid)) expect(Object.values(g.mix).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 3);
    for (const o of pipe.ood_alphas) {
      const cov = pipe.cp_alphas.map((c) => grid[key(o, c)].coverage);
      cov.slice(1).forEach((v, i) => expect(v).toBeLessThanOrEqual(cov[i]));
    }
  });
});
