const norm = (s: string) => s.toLowerCase().replace(/[_\s]+/g, "-");

/** "crack_01.jpg" -> "crack". The class must be followed by a non-letter (or end), so "cracked" is not "crack". */
export function labelFromFilename(name: string, classes: string[]): string | undefined {
  const base = norm(name.replace(/^.*[\\/]/, ""));
  for (const c of [...classes].sort((a, b) => b.length - a.length)) {
    const k = norm(c);
    if (base.startsWith(k) && !/[a-z]/.test(base.charAt(k.length))) return c;
  }
  return undefined;
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "", q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (q) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') q = false;
      else cur += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

/** labels.csv with at least `file` and `cls` columns (the repo's samples format). Unknown classes are ignored. */
export function parseLabelsCsv(text: string, classes: string[]): Map<string, string> {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/).filter((l) => l.trim());
  const out = new Map<string, string>();
  if (!lines.length) return out;
  const head = splitCsvLine(lines[0]).map((h) => h.toLowerCase());
  const fi = head.indexOf("file"), ci = head.indexOf("cls");
  if (fi < 0 || ci < 0) return out;
  const known = new Map(classes.map((c) => [norm(c), c]));
  for (const l of lines.slice(1)) {
    const cells = splitCsvLine(l);
    const raw = cells[ci] ?? "";
    const cls = classes.length ? known.get(norm(raw)) : raw || undefined;
    if (cells[fi] && cls) out.set(cells[fi], cls);
  }
  return out;
}

/** The true class for a file: labels.csv first, then the file name. Undefined until the classes are known. */
export function truthFor(name: string, csvLabel: string | undefined, classes: string[]):
  { cls: string; source: "csv" | "filename" } | undefined {
  if (!classes.length) return undefined;
  if (csvLabel) {
    const cls = classes.find((c) => norm(c) === norm(csvLabel));
    return cls ? { cls, source: "csv" } : undefined;
  }
  const cls = labelFromFilename(name, classes);
  return cls ? { cls, source: "filename" } : undefined;
}
