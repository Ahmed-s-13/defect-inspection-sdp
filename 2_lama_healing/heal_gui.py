"""Simple desktop GUI for the healing framework (Tkinter, nothing to install).

  python heal_gui.py

1. Open photo        boxes load automatically if the photo has a YOLO label file (D1 photos, or x.txt next to x.jpg)
2. Draw boxes        left-drag on the original = new box of the class chosen in the dropdown; right-click = delete
3. Run healing       LaMa for every box, a screw head put back for missing-head boxes (heal.heal_photo)
4. Save result       the healed photo as PNG

Panels: original (with boxes) | LaMa only | framework result. The log shows what happened to each box.
"""
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import heal as hf
from hcore import RAW, find_label_file, read_yolo

CLASSES = ["missing-head", "crack", "dent", "corrosion", "scratch", "paint-peel-off"]
COLOURS = {"missing-head": "#00e5ff", "crack": "#ff5252", "dent": "#ffd740", "corrosion": "#ff9100",
           "scratch": "#e040fb", "paint-peel-off": "#69f0ae"}
PANEL = 420                                                        # px per image panel


class App:
    def __init__(self, root):
        self.root = root
        root.title("Aircraft defect healing: LaMa + screw-head transplant")
        self.img = self.clean = self.out = None                    # numpy RGB photos
        self.path, self.boxes = None, []                           # boxes: [cls, x1, y1, x2, y2] in photo pixels
        self.models, self.busy, self.drag = None, False, None

        bar = ttk.Frame(root, padding=6)
        bar.pack(fill="x")
        self.b_open = ttk.Button(bar, text="Open photo", command=self.open)
        self.b_run = ttk.Button(bar, text="Run healing", command=self.run, state="disabled")
        self.b_save = ttk.Button(bar, text="Save result", command=self.save, state="disabled")
        self.b_clear = ttk.Button(bar, text="Clear boxes", command=self.clear)
        for b in (self.b_open, self.b_run, self.b_save, self.b_clear):
            b.pack(side="left", padx=3)
        ttk.Label(bar, text="  New box class:").pack(side="left")
        self.cls = tk.StringVar(value="missing-head")
        ttk.Combobox(bar, textvariable=self.cls, values=CLASSES, state="readonly", width=15).pack(side="left", padx=3)
        self.status = tk.StringVar(value="Loading models (LaMa + head scorer)...")
        ttk.Label(bar, textvariable=self.status, foreground="#555").pack(side="left", padx=12)

        panels = ttk.Frame(root, padding=(6, 0))
        panels.pack()
        self.canvas = []
        for i, title in enumerate(["Original (drag = new box, right-click = delete)", "LaMa only", "Framework result"]):
            f = ttk.Frame(panels)
            f.grid(row=0, column=i, padx=4)
            ttk.Label(f, text=title).pack()
            c = tk.Canvas(f, width=PANEL, height=PANEL, bg="#222", highlightthickness=0)
            c.pack()
            self.canvas.append(c)
        c0 = self.canvas[0]
        c0.bind("<ButtonPress-1>", self.drag_start)
        c0.bind("<B1-Motion>", self.drag_move)
        c0.bind("<ButtonRelease-1>", self.drag_end)
        c0.bind("<Button-3>", self.delete_box)

        ttk.Label(root, text="Log", padding=(8, 4, 0, 0)).pack(anchor="w")
        self.log = tk.Text(root, height=9, font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.tk_imgs = [None, None, None]
        self.events = queue.Queue()                                # worker threads -> GUI (Tk is not thread-safe)
        self.poll()
        threading.Thread(target=self.load_models, daemon=True).start()

    def poll(self):
        """Run what the worker threads queued, on the GUI thread."""
        while True:
            try:
                fn = self.events.get_nowait()
            except queue.Empty:
                break
            fn()
        self.root.after(100, self.poll)

    # ------------------------------------------------------------------ models
    def load_models(self):
        try:
            h = hf.Healer()
            msg = f"Ready ({h.dev.type.upper()}, donor bank {len(h.bank)} heads). Open a photo."
            self.events.put(lambda: (setattr(self, "models", (h.lama, h.scorer, h.bank, h.dev)), self.ready(msg)))
        except BaseException as e:                                 # noqa: BLE001  (SystemExit: missing weights)
            err = f"Model loading failed: {e}"
            self.events.put(lambda: self.ready(err, ok=False))

    def ready(self, msg, ok=True):
        self.status.set(msg)
        if ok and self.img is not None:
            self.b_run.config(state="normal")

    # ------------------------------------------------------------------ photo + boxes
    def open(self):
        f = filedialog.askopenfilename(title="Open photo", initialdir=RAW / "images" if RAW.exists() else hf.HERE,
                                       filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")])
        if not f:
            return
        self.path = Path(f)
        self.img = np.asarray(Image.open(f).convert("RGB")).copy()
        self.clean = self.out = None
        self.boxes = []
        lf = find_label_file(self.path)                            # a YOLO label file (D1 photos have one)
        if lf:
            self.boxes = [list(b) for b in read_yolo(lf, self.img.shape[1], self.img.shape[0])]
            msg = f"{len(self.boxes)} labelled box(es) loaded from {lf.name}"
        else:
            msg = "no label file: draw the defect boxes"
        self.write(f"Opened {self.path.name} ({self.img.shape[1]}x{self.img.shape[0]}): {msg}")
        self.b_save.config(state="disabled")
        self.b_run.config(state="normal" if self.models else "disabled")
        self.redraw()

    def clear(self):
        self.boxes = []
        self.redraw()

    def scale(self):
        h, w = self.img.shape[:2]
        s = PANEL / max(h, w)
        return s, (PANEL - w * s) / 2, (PANEL - h * s) / 2         # scale, x offset, y offset (image is centred)

    def to_photo(self, x, y):
        s, ox, oy = self.scale()
        return (x - ox) / s, (y - oy) / s

    def show(self, i, arr):
        c = self.canvas[i]
        c.delete("all")
        if arr is None:
            return
        s, ox, oy = self.scale()
        h, w = arr.shape[:2]
        im = Image.fromarray(arr).resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        self.tk_imgs[i] = ImageTk.PhotoImage(im)
        c.create_image(ox, oy, anchor="nw", image=self.tk_imgs[i])

    def redraw(self):
        if self.img is None:
            return
        self.show(0, self.img)
        s, ox, oy = self.scale()
        for cls, x1, y1, x2, y2 in self.boxes:
            col = COLOURS.get(cls, "white")
            self.canvas[0].create_rectangle(ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s, outline=col, width=2)
        self.show(1, self.clean)
        self.show(2, self.out)

    def drag_start(self, ev):
        if self.img is not None and not self.busy:
            self.drag = (ev.x, ev.y, self.canvas[0].create_rectangle(ev.x, ev.y, ev.x, ev.y,
                                                                      outline=COLOURS[self.cls.get()], width=2, dash=(4, 2)))

    def drag_move(self, ev):
        if self.drag:
            self.canvas[0].coords(self.drag[2], self.drag[0], self.drag[1], ev.x, ev.y)

    def drag_end(self, ev):
        if not self.drag:
            return
        x0, y0, _ = self.drag
        self.drag = None
        (a, b), (c, d) = self.to_photo(x0, y0), self.to_photo(ev.x, ev.y)
        h, w = self.img.shape[:2]
        x1, x2 = sorted((np.clip(a, 0, w), np.clip(c, 0, w)))
        y1, y2 = sorted((np.clip(b, 0, h), np.clip(d, 0, h)))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            self.boxes.append([self.cls.get(), float(x1), float(y1), float(x2), float(y2)])
            self.write(f"+ {self.cls.get()} box ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        self.redraw()

    def delete_box(self, ev):
        if self.img is None or self.busy:
            return
        x, y = self.to_photo(ev.x, ev.y)
        hits = [i for i, (_, x1, y1, x2, y2) in enumerate(self.boxes) if x1 <= x <= x2 and y1 <= y <= y2]
        if hits:                                                   # the smallest box under the cursor
            i = min(hits, key=lambda j: (self.boxes[j][3] - self.boxes[j][1]) * (self.boxes[j][4] - self.boxes[j][2]))
            self.write(f"- removed {self.boxes[i][0]} box")
            del self.boxes[i]
            self.redraw()

    # ------------------------------------------------------------------ healing
    def run(self):
        if not self.boxes:
            messagebox.showinfo("No boxes", "Draw at least one defect box first (left-drag on the original).")
            return
        self.busy = True
        for b in (self.b_open, self.b_run, self.b_save, self.b_clear):
            b.config(state="disabled")
        self.status.set("Healing...")
        threading.Thread(target=self.heal, daemon=True).start()

    def heal(self):
        try:
            lama, scorer, bank, dev = self.models
            t0 = time.time()
            clean, out, rows = hf.heal_photo(self.img, [tuple(b) for b in self.boxes], lama, scorer, bank, dev)
            secs = time.time() - t0
            self.events.put(lambda: self.done(clean, out, rows, secs))
        except Exception as e:                                     # noqa: BLE001
            err = str(e)
            self.events.put(lambda: self.done(None, None, [], 0, error=err))

    def done(self, clean, out, rows, secs, error=None):
        self.busy = False
        for b in (self.b_open, self.b_run, self.b_clear):
            b.config(state="normal")
        if error:
            self.status.set("Healing failed")
            self.write(f"ERROR: {error}")
            return
        self.clean, self.out = clean, out
        self.b_save.config(state="normal")
        heads = sum(r["route"].startswith("head") for r in rows)
        self.status.set(f"Done in {secs:.1f} s: {len(rows)} box(es), {heads} screw head(s) put back")
        for r in rows:
            extra = f", donor {r['donor']}" if r.get("donor") else ""
            extra += f", head score {r['p_fastener']}" if r.get("p_fastener") not in ("", None) else ""
            self.write(f"  box {r['box_i'] + 1} {r['cls']:15s} -> {r['route']}{extra}")
        self.redraw()

    def save(self):
        f = filedialog.asksaveasfilename(title="Save healed photo", defaultextension=".png",
                                         initialfile=f"{self.path.stem}_healed.png", filetypes=[("PNG", "*.png")])
        if f:
            Image.fromarray(self.out).save(f)
            self.write(f"Saved {f}")

    def write(self, s):
        self.log.insert("end", s + "\n")
        self.log.see("end")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
