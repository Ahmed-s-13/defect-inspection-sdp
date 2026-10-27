export interface Pair { truth: string; pred: string }

export const accuracy = (p: Pair[]) => (p.length ? p.filter((x) => x.truth === x.pred).length / p.length : NaN);

/** Wilson score interval for k successes in n trials. */
export function wilson(k: number, n: number, z = 1.96): [number, number] {
  if (!n) return [NaN, NaN];
  const ph = k / n, z2 = z * z;
  const centre = (ph + z2 / (2 * n)) / (1 + z2 / n);
  const half = (z * Math.sqrt((ph * (1 - ph)) / n + z2 / (4 * n * n))) / (1 + z2 / n);
  return [centre - half, centre + half];
}

export function confusion(p: Pair[], classes: string[]): number[][] {
  const m = classes.map(() => classes.map(() => 0));
  const ix = new Map(classes.map((c, i) => [c, i]));
  for (const { truth, pred } of p) {
    const r = ix.get(truth), c = ix.get(pred);
    if (r !== undefined && c !== undefined) m[r][c]++;
  }
  return m;
}

export function perClass(p: Pair[], classes: string[]) {
  const m = confusion(p, classes);
  return classes.map((cls, i) => {
    const tp = m[i][i];
    const support = m[i].reduce((a, b) => a + b, 0);
    const predicted = m.reduce((a, row) => a + row[i], 0);
    const precision = predicted ? tp / predicted : 0;
    const recall = support ? tp / support : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { cls, precision, recall, f1, support, predicted };
  });
}

/** Macro-F1 over the classes that appear as truth or prediction (scikit-learn's default). */
export function macroF1(p: Pair[], classes: string[]): number {
  const rows = perClass(p, classes).filter((r) => r.support > 0 || r.predicted > 0);
  return rows.length ? rows.reduce((a, r) => a + r.f1, 0) / rows.length : NaN;
}

/** The answer the inspector saw: architecture A's set, or the top class for a model without it. */
interface Answered { state: string; answer: string[]; top: string }
const shownAnswer = (r: Answered) => (r.state === "unrated" ? [r.top] : r.answer);

/** True when an answer was shown and the true class is not in it. A gate's refusal is not a miss. */
export const answerMisses = (r: Answered, truth: string) => {
  const a = shownAnswer(r);
  return a.length > 0 && !a.includes(truth);
};

/** Over labelled images: how many got an answer, how many answers held the truth, and how many were a single
 *  wrong class (shown with no warning). */
export function answerStats(rows: { r: Answered; truth: string }[]) {
  const answered = rows.filter(({ r }) => shownAnswer(r).length > 0);
  return {
    answered: answered.length,
    holds: answered.filter(({ r, truth }) => shownAnswer(r).includes(truth)).length,
    silent: answered.filter(({ r, truth }) => shownAnswer(r).length === 1 && !shownAnswer(r).includes(truth)).length,
  };
}
