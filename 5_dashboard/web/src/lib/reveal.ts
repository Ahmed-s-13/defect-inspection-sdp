/** Whether a section should be shown after a scroll step.
 *  top: the section's top edge in viewport px; dir: scroll delta (+ down, − up, 0 = no scroll);
 *  vh: viewport height; show / hide: the lines (fractions of vh) for playing in and dissolving. */
export function nextShown(prev: boolean, top: number, dir: number, vh: number, show: number, hide: number): boolean {
  if (dir > 0 && top < vh * show) return true;   // scrolling down: plays in as it rises past the show line
  if (dir < 0 && top > vh * hide) return false;  // scrolling up: dissolves while still on screen
  if (dir === 0) return top < vh * show;         // first paint, resize, tab switch
  return prev;
}
