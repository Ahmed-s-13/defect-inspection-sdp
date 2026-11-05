import { describe, expect, it } from "vitest";
import { accuracy, answerMisses, answerStats, confusion, macroF1, perClass, wilson } from "./metrics";

const C = ["a", "b", "c"];
const P = [
  { truth: "a", pred: "a" }, { truth: "a", pred: "b" }, { truth: "b", pred: "b" },
  { truth: "b", pred: "b" }, { truth: "c", pred: "a" },
];

describe("metrics", () => {
  it("accuracy", () => { expect(accuracy(P)).toBeCloseTo(0.6); expect(accuracy([])).toBeNaN(); });
  it("wilson interval", () => {
    const [lo, hi] = wilson(26, 30);
    expect(lo).toBeCloseTo(0.7032, 3);
    expect(hi).toBeCloseTo(0.9469, 3);
    expect(wilson(0, 0)).toEqual([NaN, NaN]);
  });
  it("confusion rows are truth", () => {
    expect(confusion(P, C)).toEqual([[1, 1, 0], [0, 2, 0], [1, 0, 0]]);
  });
  it("per class and macro-F1 over labels present (sklearn convention)", () => {
    const pc = perClass(P, C);
    expect(pc[0]).toMatchObject({ cls: "a", precision: 0.5, recall: 0.5, support: 2 });
    expect(pc[1].f1).toBeCloseTo(0.8);
    expect(macroF1(P, C)).toBeCloseTo((0.5 + 0.8 + 0) / 3);
    expect(macroF1([{ truth: "a", pred: "a" }], C)).toBe(1);
  });
});

describe("architecture A answers", () => {
  const r = (state: string, answer: string[], top = answer[0] ?? "crack") => ({ state, answer, top });
  it("a refusal is never a miss; a set that holds the truth is not either", () => {
    expect(answerMisses(r("retake", [], "dent"), "crack")).toBe(false);
    expect(answerMisses(r("ambiguous", ["dent", "crack"]), "crack")).toBe(false);
    expect(answerMisses(r("detected", ["dent"]), "crack")).toBe(true);
    expect(answerMisses(r("unrated", [], "dent"), "crack")).toBe(true); // legacy model: its top class is the answer
  });
  it("counts answers, answers holding the truth, and single wrong classes", () => {
    const rows = [
      { r: r("detected", ["crack"]), truth: "crack" }, { r: r("detected", ["dent"]), truth: "crack" },
      { r: r("ambiguous", ["dent", "crack"]), truth: "crack" }, { r: r("cannot_assess", []), truth: "crack" },
    ];
    expect(answerStats(rows)).toEqual({ answered: 3, holds: 2, silent: 1 });
  });
});
