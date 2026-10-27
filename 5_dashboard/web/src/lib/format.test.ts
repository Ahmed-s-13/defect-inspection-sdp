import { describe, expect, it } from "vitest";
import { className, int, pct, shortClass } from "./format";

describe("format", () => {
  it("percent and integers", () => {
    expect(pct(0.80446)).toBe("80.4%");
    expect(pct(NaN)).toBe("—");
    expect(int(4255)).toBe("4,255");
  });
  it("class names in plain words", () => {
    expect(className("missing-head")).toBe("Missing head");
    expect(className("paint-peel-off")).toBe("Paint peel-off");
    expect(shortClass("paint-peel-off")).toBe("Peel-off");
  });
});
