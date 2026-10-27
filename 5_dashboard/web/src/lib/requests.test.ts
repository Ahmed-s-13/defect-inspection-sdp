import { describe, expect, it } from "vitest";
import { chunkWithOffsets, latestGate } from "./requests";

describe("latestGate", () => {
  it("only the newest request per key may write its reply", () => {
    const g = latestGate();
    const first = g.next("i1");
    const second = g.next("i1");
    expect(g.isCurrent("i1", first)).toBe(false);   // older reply arriving late is dropped
    expect(g.isCurrent("i1", second)).toBe(true);
    expect(g.isCurrent("i2", g.next("i2"))).toBe(true); // keys are independent
  });
});

describe("chunkWithOffsets", () => {
  it("splits uploads into chunks and keeps global indices", () => {
    const parts = chunkWithOffsets([..."abcde"], 2);
    expect(parts).toEqual([{ offset: 0, items: ["a", "b"] }, { offset: 2, items: ["c", "d"] }, { offset: 4, items: ["e"] }]);
    expect(chunkWithOffsets([], 64)).toEqual([]);
  });
});
