import { describe, expect, it } from "vitest";
import { labelFromFilename, parseLabelsCsv } from "./labels";

const C = ["corrosion", "crack", "dent", "missing-head", "paint-peel-off", "scratch"];

describe("labelFromFilename", () => {
  it("reads a class prefix", () => {
    expect(labelFromFilename("crack_01.jpg", C)).toBe("crack");
    expect(labelFromFilename("Dent-3.PNG", C)).toBe("dent");
    expect(labelFromFilename("missing-head_05.jpg", C)).toBe("missing-head");
    expect(labelFromFilename("missing_head_02.jpg", C)).toBe("missing-head");
    expect(labelFromFilename("paint peel off 7.jpg", C)).toBe("paint-peel-off");
    expect(labelFromFilename("folder/scratch12.jpg", C)).toBe("scratch");
  });
  it("refuses partial or unknown prefixes", () => {
    expect(labelFromFilename("cracked_01.jpg", C)).toBeUndefined();
    expect(labelFromFilename("IMG_2231.jpg", C)).toBeUndefined();
    expect(labelFromFilename("dents.jpg", C)).toBeUndefined();
  });
});

describe("parseLabelsCsv", () => {
  it("handles BOM, CRLF, extra columns, quotes and unknown classes", () => {
    const text = "﻿file,cls,source_crop\r\ncrack_01.jpg,crack,x\r\n\"b, c.jpg\",dent,y\r\nz.jpg,bird,q\r\n";
    const m = parseLabelsCsv(text, C);
    expect(m.get("crack_01.jpg")).toBe("crack");
    expect(m.get("b, c.jpg")).toBe("dent");
    expect(m.has("z.jpg")).toBe(false);
  });
  it("returns empty without file/cls columns", () => {
    expect(parseLabelsCsv("a,b\n1,2\n", C).size).toBe(0);
  });
});

import { truthFor } from "./labels";

describe("truthFor", () => {
  it("prefers labels.csv, then the file name, and resolves only once classes are known", () => {
    expect(truthFor("IMG_1.jpg", "dent", C)).toEqual({ cls: "dent", source: "csv" });
    expect(truthFor("crack_01.jpg", undefined, C)).toEqual({ cls: "crack", source: "filename" });
    expect(truthFor("crack_01.jpg", "Missing_Head", C)).toEqual({ cls: "missing-head", source: "csv" });
    expect(truthFor("crack_01.jpg", undefined, [])).toBeUndefined();  // model still loading: try again later
    expect(truthFor("IMG_1.jpg", "bird", C)).toBeUndefined();
  });
});
