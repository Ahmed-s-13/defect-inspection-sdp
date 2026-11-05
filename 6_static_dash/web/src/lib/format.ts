export const pct = (x: number, d = 1) => (Number.isFinite(x) ? `${(x * 100).toFixed(d)}%` : "—");
export const int = (n: number) => n.toLocaleString("en-US");
export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const NAMES: Record<string, string> = { "missing-head": "Missing head", "paint-peel-off": "Paint peel-off" };
const SHORT: Record<string, string> = { "missing-head": "Missing head", "paint-peel-off": "Peel-off", corrosion: "Corrosion" };
export const className = (c: string) => NAMES[c] ?? cap(c.replace(/-/g, " "));
export const shortClass = (c: string) => SHORT[c] ?? className(c);
const COL: Record<string, string> = { corrosion: "Corr.", "missing-head": "Head", "paint-peel-off": "Peel", scratch: "Scratch" };
/** Column headers in tight matrices; the full name stays in the row header and the cell title. */
export const colClass = (c: string) => COL[c] ?? shortClass(c);
