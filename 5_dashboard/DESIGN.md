---
name: Defect Test Bench
description: A local inspection instrument for the project's main aircraft-surface defect classifier.
colors:
  ground: "#F2F1EE"
  sheet: "#FAFAF9"
  ink: "#16181B"
  ink-2: "#464A51"
  ink-3: "#64686F"
  rule: "#DEDCD5"
  rule-strong: "#C8C5BC"
  bar: "#BDB9AF"
  well: "#6B6E72"
  well-ink: "#F2F1EE"
  detected: "#A4262C"
  detected-tint: "#F5E3E2"
  clear: "#2F7A4C"
  clear-tint: "#E2EFE6"
  verify: "#C99A06"
  verify-ink: "#6B5000"
  verify-tint: "#FAF0D0"
  unfamiliar: "#5B6573"
  unfamiliar-tint: "#E5E8EC"
  unrated: "#64686F"
  fault: "#A4262C"
  control: "#0E7C86"
  control-ink: "#0A5960"
  control-tint: "#DCEFF0"
typography:
  hero-figure:
    fontFamily: "B612, Atkinson Hyperlegible Next, sans-serif"
    fontSize: "calc(56px * 1.6)"
    fontWeight: 700
    lineHeight: 0.95
    letterSpacing: "-0.03em"
  display:
    fontFamily: "B612, Atkinson Hyperlegible Next, sans-serif"
    fontSize: "56px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.02em"
  figure:
    fontFamily: "B612, Atkinson Hyperlegible Next, sans-serif"
    fontSize: "34px"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "-0.01em"
    fontFeature: "tnum"
  headline:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 700
  title:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 400
  body:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
  micro:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
rounded:
  hairline: "2px"
  thumb: "3px"
  base: "6px"
  menu: "8px"
  dot: "50%"
spacing:
  xs: "8px"
  sm: "16px"
  gap: "24px"
  pad: "32px"
  topbar: "56px"
components:
  button:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.base}"
    padding: "0 14px"
    height: "36px"
  button-primary:
    backgroundColor: "{colors.control}"
    textColor: "#FFFFFF"
    rounded: "{rounded.base}"
    padding: "0 14px"
    height: "36px"
  button-primary-hover:
    backgroundColor: "{colors.control-ink}"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.ink-3}"
  button-well:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.ink}"
    rounded: "{rounded.base}"
    padding: "0 18px"
    height: "40px"
  chip-toggle:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
    padding: "0 12px"
    height: "34px"
  chip-toggle-on:
    backgroundColor: "{colors.control}"
    textColor: "#FFFFFF"
  segment:
    backgroundColor: "transparent"
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
    padding: "0 12px"
    height: "32px"
  segment-selected:
    backgroundColor: "{colors.control}"
    textColor: "#FFFFFF"
  tab:
    textColor: "{colors.ink-3}"
    padding: "0 16px"
  tab-selected:
    textColor: "{colors.ink}"
  tile:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
  image-well:
    backgroundColor: "{colors.well}"
    textColor: "{colors.well-ink}"
    rounded: "{rounded.base}"
  caveat-note:
    backgroundColor: "{colors.unfamiliar-tint}"
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
    padding: "12px 14px"
  softmax-track:
    backgroundColor: "{colors.rule}"
    rounded: "{rounded.hairline}"
    height: "10px"
---

# Design System: Defect Test Bench

## Overview

**Creative North Star: "The Inspection Lens"**

The app is a borescope instrument without the props. A mid-grey well holds the untouched photo as the stage; a plain bracketed reticle marks exactly the square the model judges; the readout beside it states the verdict in the wearable's own state words and then shows the evidence. The chrome around it is warm-neutral paper with 1px instrument rules. It is quiet so that the photo and the verdict carry the room.

Density is that of a working bench: compact labels, tabular figures, sorted bars, ruled lists. Presentation mode keeps the same layout and enlarges it for a projected room. Type scales up, gaps widen, and the signature motion slows. The world deliberately refuses the category default of an upload card beside a probability bar chart in a grid of cards. Surfaces are flat sheets divided by rules, not floating cards.

Colour is a signal, never decoration. Only the answer state carries hue (crimson, green, ochre, slate), teal marks the things you can operate, and the classes get no colour at all.

**Key Characteristics:**
- Mid-grey image well holding untinted photos; the reticle is the model's literal field of view.
- Warm-neutral ground, near-white sheets, near-black ink, 1px rules; no elevation.
- Hue reserved for confidence state and controls; classes are never colour-coded.
- B612 for every figure and the verdict; Atkinson Hyperlegible Next for everything you read as UI.
- Motion only through Motion (framer), only transform and opacity, and values morph from their previous state.
- A single scale switch (Presentation mode) changes type, gap and timing tokens, not layout.

## Colors

A warm-neutral instrument palette with three state hues and one control hue, each assigned to exactly one job.

### Primary
- **Instrument Teal** (control): the only colour for operable things. Selected tab underline, primary button, the pressed Presentation toggle, selected segment, focus ring, caret, range-slider accent, chart cursor. **Deep Teal** (control-ink) is its hover and its text-on-tint shade, and **Teal Wash** (control-tint) is the selection and clear-filter background.

### Secondary (confidence states)
- **Signal Crimson** (detected): Defect detected. Used for the verdict state word, the reticle frame and label once the verdict lands, the top softmax bar, and the state glyph. System fault (fault) shares this value and is told apart by its glyph and words. **Crimson Wash** (detected-tint) is the fault banner background.
- **Clear Green** (clear): No defect, architecture A's single "no-defect" answer (gc7's 7th class). Verdict word, reticle, glyph (a ticked square), the bar of the kept class. **Green Wash** (clear-tint) is its background tint.
- **Caution Ochre** (verify): Ambiguous and Unclear, as a *fill* only (reticle frame, top bar, glyph) with ink on top. Text in this state uses **Deep Ochre** (verify-ink), because the ochre itself is not legible as text on the ground. **Ochre Wash** (verify-tint) is its background tint.
- **Instrument Slate** (unfamiliar): Cannot assess. Verdict word, reticle, top bar, glyph. **Slate Wash** (unfamiliar-tint) backs the always-visible caveat under the answer.
- **Uncalibrated Grey** (unrated): Not calibrated. It shares the ink-3 value and uses a dashed glyph.

### Neutral
- **Bench Ground** (ground): the page, the top bar, and the light text inside the well. A deliberate step from the contract's #F3F2EE: the original read as yellow-cream under lint, so the ground was pulled cooler.
- **Sheet** (sheet): buttons, tiles, the summary strip, the popover and confusion-matrix cells. One step lighter than ground, so sheets read as paper laid on the bench.
- **Ink** (ink): primary text, the current-model figure in charts, the filled confusion-matrix cells.
- **Ink Two** (ink-2) and **Ink Three** (ink-3): secondary text and the quietest labels (axis ticks, sub-figures, captions).
- **Rule** (rule) and **Rule Strong** (rule-strong): 1px hairlines; the strong rule tops panels and outlines buttons.
- **Baseline Stone** (bar): the neutral data colour for non-top bars, confidence intervals, the previous-model dots and the coverage line.
- **Well Grey** (well): the image well and the background behind thumbnails. Its text colour (well-ink) is the ground value.

### Named Rules
**The Classes Carry No Hue Rule.** Crack, dent, corrosion, scratch, missing-head and paint-peel-off are always named in ink. Hue on a class would read as a state.

**The State Owns The Hue Rule.** Crimson, ochre and slate appear only where a confidence state is being reported (verdict, reticle, top bar, glyph, state badge). Never use them for emphasis, headings or charts of aggregate metrics.

**The Teal Is A Handle Rule.** If it is teal you can press, drag, focus or select it. Nothing static is teal.

**The Ochre Never Speaks Rule.** Caution Ochre is a fill under ink. Text in that state is set in Deep Ochre.

## Typography

**Figure Font:** B612 (with Atkinson Hyperlegible Next, sans-serif)
**UI Font:** Atkinson Hyperlegible Next (with system-ui, sans-serif)

**Character:** B612 was drawn for cockpit displays and gives every number and the verdict an instrument voice. Atkinson Hyperlegible Next keeps labels unambiguous when read from the back of a projected room. Both are bundled offline (B612 400/700; Atkinson 400/500/700).

### Hierarchy
- **Hero figure** (B612 700, display × 1.6, 0.95, -0.03em): the single headline metric on the Model view.
- **Display** (B612 700, 56px, 1, -0.02em): the predicted class in the verdict.
- **Figure** (B612, 34px, ~1.05–1.15): the verdict state word (700, -0.01em), summary-strip figures (400), and secondary metrics. The empty-well invitation also uses this size, set in the UI face.
- **Headline** (Atkinson 700, 24px): view titles.
- **Title** (Atkinson, 18px): panel and readout titles, headline sentences.
- **Body** (Atkinson 400, 16px, 1.5): running text; prose capped at 46–70ch depending on context.
- **Label** (Atkinson, 14px): bar rows, facts lists, buttons in the top bar, segments.
- **Micro** (Atkinson, 12px): captions, axis scales, provenance, tile truth lines.

Presentation mode rescales the same roles: display 88, figure 48, headline 32, title 24, body 20, label 17, micro 15. Below 720px, display drops to 40 and figure to 28.

### Named Rules
**The Figures Are Instrument Readings Rule.** Every number (percentages, counts, ms, similarities, matrix values) and the verdict are set in B612 with tabular numerals. Prose and labels are never set in B612.

**The Sentence Case Rule.** No uppercase labels and no tracking added for labels. Letter-spacing is only ever negative, and only on display-scale B612.

## Layout

The shell is a sticky 56px top bar on a three-column grid: brand on the left, tabs centred, model chip and Presentation toggle on the right. Content sits in a canvas padded 32px horizontally with a 24px gap rhythm (16/16 on small screens, gap 32 in Presentation mode).

- **Inspect:** two columns, the well at 2fr and the readout at minmax(340px, 1fr). The well fills the viewport height under the bar, and its file bar runs beneath it. Below 960px it stacks, and the well becomes a square up to 520px.
- **Batch:** a head row (title and actions), a four-cell summary strip, then a thumbnail grid (auto-fill, 150px minimum) beside a 380px side column. That column stacks below 1100px, and the grid drops to two columns below 720px.
- **Model:** a ruled sheet. A headline pair (hero figure and facts) sits over pairs of panels, each opened by a 1px strong rule. Vertical rhythm is double the gap. Pairs stack below 1000px.

Internal rhythm is 4/8/12/16/20 px. Readout sections are separated by 20px gaps and 1px rules, not by boxes.

## Elevation & Depth

Flat. Depth comes from tone: ground, then sheet a step lighter, then the dark well as a recessed stage, plus 1px rules. No surface lifts on hover. Hover changes a border from rule to ink-3.

### Shadow Vocabulary
- **Reticle legibility ring** (`box-shadow: 0 0 0 1px rgba(22, 24, 27, .45)`): a spread-only dark hairline that keeps the light reticle visible on pale photos. It works as an outline, not elevation.

### Named Rules
**The Ruled Sheet Rule.** Group content with a 1px rule and space, not a filled card. A bordered sheet is reserved for things you act on or scan as a set (buttons, tiles, the summary strip).

## Shapes

One modest radius (6px) for buttons, chips, tiles, the well, notes and banners. Thumbnails inside the well use 3px, and track and view thumbnails use 2px. Dots and chart markers are full circles. The reticle is square-cornered with four 4px L-brackets at its corners. Its label tab hangs from the top edge with only the lower corners rounded (3px). Drop targets show a 2px dashed outline only while something is dragged over them.

## Components

### Buttons
Restrained, bench-tool buttons. They respond with a border shift and a slight press.
- **Shape:** gently rounded (6px), 36px tall, 1px rule-strong border on sheet.
- **Primary:** Instrument Teal fill, white bold text. Hovering darkens it to Deep Teal.
- **Hover / Focus / Active:** border moves to ink-3 on hover. Focus shows a 2px teal ring offset 2px. Active scales to 0.97. Disabled drops to 45% opacity.
- **Quiet:** no border or fill, ink-3 text.
- **On the well:** the empty well's primary action is a ground-coloured 40px button with ink text that turns white on hover. Sample chips in the well are translucent dark with a 40px thumbnail.

### Chips and Toggles
- **Model chip / Presentation toggle:** 34px sheet chips with label-size text. The model chip carries a status dot (ink = ready, crimson = fault, stone = waiting). Presentation shows its keyboard key in a small outlined key-cap and turns teal when pressed.
- **Segments (class filter):** 32px transparent outlined segments in ink-2. The selected one fills teal. The "clear" segment uses Teal Wash with a teal border.

### Cards / Containers
- **Tiles (Batch):** sheet, 1px rule border, 6px radius, a square image on the well grey, label-size meta beneath. A wrong prediction is marked by a bold crimson ring (1px border plus a 2px inset outline) and a bold ink truth line; the user asked for red here so mistakes stand out at a glance.
- **Summary strip:** one sheet split into four cells by 1px rules. B612 figures sit over ink-3 labels.
- **Model panels:** no container. A 1px strong rule on top, a title and a sub-line.
- **Notes:** the caveat is a Slate Wash block with ink-2 text. The fault banner is a Crimson Wash block with ink text and a glyph.

### Navigation
Text tabs in ink-3, weight 500, which turn ink on hover and when selected. A 2px teal underline slides between tabs as a shared layout element (spring 500/40). On small screens the tabs move to a full-width 44px row beneath the brand.

### The Reticle and Fan (signature)
A square drawn with a 2px frame in well-ink and four heavy L-brackets that resize it. It drags with a fast spring (700/48) and jumps instantly while dragged directly. A tab on its top edge reads the model's view size. When a verdict lands, frame, brackets and tab cross-fade (0.2s) to the state hue. They clear at once on a new run. While inference runs, eight rotated and flipped copies of the crop fan out of the reticle into the well (spring 260/28, 30ms stagger). When the answer arrives they fold into the eight-view strip in the readout through shared layout ids. Presentation mode slows this to spring 110/18 with a 70ms stagger.

### Verdict Readout (signature)
The state glyph and state word appear in B612 at figure size in the state colour, with the class in ink at display size underneath. The verdict text appears instantly. A reliability sentence then cross-fades in (0.15s). Six softmax rows follow, sorted, each with a 10px stone track: the top row is bold ink with its fill in the state hue, and the other rows stay ink-3 on stone. Rows re-sort with a position spring (500/42) and figures morph from their previous value (spring 220/30, never counting up from zero). Then come the eight-view strip, a facts list on an 11em label column, and the always-visible caveat.

### State Glyphs
Small 16px inline SVG marks drawn in the state colour, one shape per state so state never depends on hue alone: a filled square in a frame (detected), a filled triangle with an ink mark (possible), a slashed circle (cannot assess), a dashed circle (not calibrated), and an outlined triangle (system fault).

## Do's and Don'ts

### Do:
- **Do** keep photos untinted on the mid-grey well. The reticle is the only overlay.
- **Do** report state with three signals together: hue, glyph shape and the wearable's state words ("Defect detected", "No defect", "Ambiguous", "Bad image", "Cannot assess").
- **Do** set every figure in B612 with tabular numerals, and morph changed figures from their previous value.
- **Do** animate only transform and opacity through Motion. Under prefers-reduced-motion, figures jump and CSS transitions collapse.
- **Do** separate groups with 1px rules and space, and outline actionable sheets with a 1px rule.
- **Do** route Presentation mode through the type, gap and timing tokens rather than a separate layout.

### Don't:
- **Don't** give a defect class a colour, or use a state hue for anything except reporting that state.
- **Don't** set text in Caution Ochre. Use Deep Ochre.
- **Don't** use teal on anything that cannot be operated.
- **Don't** wrap content in filled, shadowed cards, or put an upload card beside a probability chart in a card grid.
- **Don't** add drop shadows for elevation. The reticle's 1px ring is an outline, not a shadow.
- **Don't** add uppercase tracked labels above headings or figures.
- **Don't** count figures up from zero on mount, and don't animate the verdict text.
