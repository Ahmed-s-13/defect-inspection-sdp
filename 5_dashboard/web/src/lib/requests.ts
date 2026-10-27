/** Per-key request tokens: a reply may write only if no newer request for the same key has started. */
export function latestGate() {
  const latest = new Map<string, number>();
  let seq = 0;
  return {
    next(key: string) { const t = ++seq; latest.set(key, t); return t; },
    isCurrent(key: string, token: number) { return latest.get(key) === token; },
  };
}

/** Split a list into chunks, remembering where each chunk starts in the whole list. */
export function chunkWithOffsets<T>(items: T[], size: number): { offset: number; items: T[] }[] {
  const out: { offset: number; items: T[] }[] = [];
  for (let i = 0; i < items.length; i += size) out.push({ offset: i, items: items.slice(i, i + size) });
  return out;
}
