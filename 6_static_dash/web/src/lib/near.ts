/** static dash: whether an element is within about a screen of the viewport. Scroll-linked scenes and reveals skip
 *  their per-frame work (layout reads, SVG and style writes) while they are far away, so scrolling one part of a
 *  long page doesn't make a phone redo every other part. `onChange` runs when it comes near or goes away, so a scene
 *  can settle in its final state on the way out. */
export function watchNear(el: Element, onChange: () => void, margin = "100% 0px") {
  const s = { near: true };
  if (typeof IntersectionObserver === "undefined") return { get near() { return true; }, stop: () => undefined };
  const io = new IntersectionObserver((entries) => { s.near = entries[entries.length - 1].isIntersecting; onChange(); }, { rootMargin: margin });
  io.observe(el);
  return { get near() { return s.near; }, stop: () => io.disconnect() };
}
