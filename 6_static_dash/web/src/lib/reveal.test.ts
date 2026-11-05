import { describe, expect, it } from "vitest";
import { nextShown } from "./reveal";

const VH = 900, SHOW = 0.85, HIDE = 0.55;
const step = (prev: boolean, top: number, dir: number) => nextShown(prev, top, dir, VH, SHOW, HIDE);

describe("nextShown (direction-aware section reveal)", () => {
  it("first paint: shown only if the section is already up in view", () => {
    expect(step(false, 2000, 0)).toBe(false);
    expect(step(false, 300, 0)).toBe(true);
  });
  it("scrolling down plays it in once its top rises above 85% of the screen", () => {
    expect(step(false, 800, +60)).toBe(false);
    expect(step(false, 760, +60)).toBe(true);
  });
  it("scrolling up dissolves it while it is still on screen (top below 55%, so 45% of the screen shows it)", () => {
    expect(step(true, 480, -60)).toBe(true);
    const top = 500;                     // first position past the 55% line (495px)
    expect(step(true, top, -60)).toBe(false);
    expect(top).toBeLessThan(VH);        // the section is on screen when it starts to leave
  });
  it("a section scrolled past at the top stays shown when scrolling up through it", () => {
    expect(step(true, -400, -60)).toBe(true);
  });
  it("scrolling down again replays it", () => {
    expect(step(false, 600, +60)).toBe(true);
  });
});
