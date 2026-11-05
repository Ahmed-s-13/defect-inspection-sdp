# Static dash

Static dash is a copy of the dashboard (`5_dashboard`) built to be hosted as a plain static website, so people at the
presentation can open it from a QR code. No model or server runs: every number and answer on it was saved from the
real dashboard's server beforehand. We never change `5_dashboard` for it.

What is different from the dashboard:

- **Top bar**: the SAQR logo and "SAQR AVI" in the logo's maroon (icons: `web/public/brand/saqr-icon-*.png` and
  `web/public/favicon.ico`; renamed from favicon-*.png so browsers drop the old cached icon). There is no model chip, no model
  switching and no presentation mode.
- **Inspect** starts empty and offers 7 sample photos, one per class, with the answers the model gave them. There is
  no upload, drop or paste, and the square is fixed at the centre. The photos are static dash's own picks
  (`samples/inspect/`), chosen by eye for how clearly they show each defect (see below).
- **Batch** runs only the 35 random test samples. Their saved answers come in one by one, as the real stream does.
  There is no Add images, Add folder or Export CSV.
- **Uncertainty** tells the same story in fewer words. The seven experiment tables at the bottom are one section of
  small charts, one per design choice (which check, how strict, how to combine, how to calibrate, family weighting,
  the miss guard, a bigger calibration set). On a phone the page is about a third shorter.
- **Helmet** on a phone flows down the page: the title, a helmet as wide as the screen, the open part's card, then the
  parts one above the other. Sideways drags turn the helmet; up and down scrolls.
- **Model** and **Data** are unchanged.
- On phones, the landing page's sketch, title and falcon scale with the screen's height as well as its width;
  Uncertainty's "Follow a crop" stages are one thin progress row with the stage's name and buttons under it; and
  lists of numbers sit two or three across instead of one per line.
- On phones the logo and name sit centred above the six tabs, which share the row evenly, and every page fits the
  screen width. The phone rules are at the end of `web/src/styles/shell.css` and `helmet.css`.

## Inspect photos

`scripts/inspect_candidates.py` makes contact sheets of d1_gc7_fix test crops (never trained on), one per photo
group, in `scripts/candidates/`. We chose from them by eye and listed the choices, best first, in
`scripts/pick_inspect.py`. That script uses a choice only if the model answers it with its true class alone and the
unfamiliar-input check passes comfortably (the same rule as the dashboard's own picks); otherwise it tries the next.
Because they are picked for clarity, these photos say nothing about accuracy; Batch's samples are the random ones.

```bash
cd "Project Code/6_static_dash" && ../../Experiments/.venv/bin/python scripts/inspect_candidates.py && ../../Experiments/.venv/bin/python scripts/pick_inspect.py
```

Then re-export (below) so the site carries the new photos and their answers.

## Layout

```
export_static_data.py     saves the server's answers into web/public/static-data/
web/                      the site (React + Vite); a copy of 5_dashboard/web
  src/api/client.ts       reads static-data/ in place of the server's /api
  public/static-data/     models.json, report.json, samples, predictions.json, crops (about 2 MB)
```

## Re-export the saved answers

Do this whenever the main model, its calibration or the samples change. Use any Python that can run `5_dashboard`'s
server; it only reads from it.

```bash
cd "Project Code/6_static_dash" && ../../Experiments/.venv/bin/python export_static_data.py
```

## Build

```bash
cd "Project Code/6_static_dash/web" && npm install && npm run build
```

The site is `web/dist/`. All paths are relative (`base: "./"`), so the folder works from any host and any subpath.
To try it locally:

```bash
cd "Project Code/6_static_dash/web/dist" && python3 -m http.server 8799
```

Then open http://localhost:8799/.

## Hosting

Not done yet. Any static host can serve `web/dist/` as it is (for example GitHub Pages, Netlify or Cloudflare Pages).
Once it has a URL, we make the QR code from it.
