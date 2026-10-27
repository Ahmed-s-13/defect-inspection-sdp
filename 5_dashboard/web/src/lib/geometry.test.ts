import { describe, expect, it } from "vitest";
import { clampSquare, defaultSquare } from "./geometry";

describe("geometry", () => {
  it("default square matches the server (integer centre square)", () => {
    expect(defaultSquare(300, 200)).toEqual({ left: 50, top: 0, side: 200 });
    expect(defaultSquare(50, 81)).toEqual({ left: 0, top: 15, side: 50 });
  });
  it("clamps inside the image with a minimum side", () => {
    expect(clampSquare({ left: -40, top: -10, side: 100 }, 300, 200)).toEqual({ left: 0, top: 0, side: 100 });
    expect(clampSquare({ left: 250, top: 150, side: 100 }, 300, 200)).toEqual({ left: 200, top: 100, side: 100 });
    expect(clampSquare({ left: 10, top: 10, side: 5000 }, 300, 200).side).toBe(200);
    expect(clampSquare({ left: 10, top: 10, side: 2 }, 300, 200).side).toBe(16);
  });
});
