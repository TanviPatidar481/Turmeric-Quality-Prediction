"""Build powder-only crop batches from the phone photos.

Each source photo is white-balanced against the paper visible around the
spread, then three square sections are cut from *inside* the powder mask:

    batch_1/  centre sections
    batch_2/  left-of-centre sections
    batch_3/  right-of-centre sections

Every batch has the same six class folders and 100 images per class, so a
batch is 600 images and the three batches total 1,800.

Important: the batches are three views of the same 600 source photos, not
three independent samples.  Never use batch_1 for training and batch_2 for
testing; split by the `spread` column in manifest.csv instead.

Usage:
    python make_crops.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prepare_crop import masks, wb_gains  # noqa: E402

SRC = HERE / "tmp_newset_1600"
TIMES = HERE / "newset_creation.txt"
OUT = HERE / "crop_dataset"
CROP_SIZE = 384
PER_CLASS = 100
OFFSETS = (0.0, -0.20, 0.20)  # centre, left, right (fraction of bbox width)
MIN_POWDER_FRAC = 0.97
BASE_FRACTION = 0.40  # square side as a fraction of the shorter bbox side


def capture_times() -> dict[str, tuple[str, datetime]]:
    times: dict[str, tuple[str, datetime]] = {}
    cur = None
    for line in TIMES.read_text().splitlines():
        line = line.strip()
        if line.lower().endswith((".heic", ".jpg", ".jpeg", ".png")):
            cur = Path(line)
        elif line.startswith("creation:") and cur:
            times[cur.stem] = (
                cur.parent.name,
                datetime.strptime(line.split(":", 1)[1].strip(), "%Y:%m:%d %H:%M:%S"),
            )
            cur = None
    return times


def spread_ids(times: dict[str, tuple[str, datetime]], gap_s: float = 6.0) -> dict[str, str]:
    """Assign each source file to a capture burst (one physical spread)."""
    by_class: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    for stem, (cls, t) in times.items():
        by_class[cls].append((stem, t))
    out: dict[str, str] = {}
    for cls, items in by_class.items():
        items.sort(key=lambda x: x[1])
        g = 0
        for i, (stem, t) in enumerate(items):
            if i and (t - items[i - 1][1]).total_seconds() > gap_s:
                g += 1
            out[stem] = f"{cls}#{g}"
    return out


def largest_powder(powder: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(powder)
    if n == 0:
        return powder
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def best_square(comp: np.ndarray, cx: int, cy: int, side: int) -> tuple[int, int, int, float] | None:
    """Largest square near (cx, cy) whose pixels are almost all powder."""
    h, w = comp.shape
    s = side
    while s >= max(48, side // 3):
        x = int(round(cx - s / 2))
        y = int(round(cy - s / 2))
        x = max(0, min(x, w - s))
        y = max(0, min(y, h - s))
        frac = float(comp[y:y + s, x:x + s].mean())
        if frac >= MIN_POWDER_FRAC:
            return x, y, s, frac
        s = int(s * 0.9)
    return None


def distance_square(
    comp: np.ndarray, dist: np.ndarray, cx: int, cy: int, side: int
) -> tuple[int, int, int, float]:
    """Snap to the deepest point near (cx, cy) and take the largest safe square."""
    h, w = comp.shape
    half = max(60, side // 2)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    sub = dist[y0:y1, x0:x1]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    py, px = y0 + int(iy), x0 + int(ix)
    s = max(48, min(side, int(np.sqrt(2) * float(dist[py, px]) * 0.90)))
    x = max(0, min(px - s // 2, w - s))
    y = max(0, min(py - s // 2, h - s))
    return x, y, s, float(comp[y:y + s, x:x + s].mean())


def make_crops(path: Path) -> tuple[list[Image.Image], list[dict]] | None:
    im = Image.open(path).convert("RGB")
    rgb = np.asarray(im).astype(np.float32)
    powder, paper = masks(rgb)
    if int(paper.sum()) < 500 or int(powder.sum()) < 1000:
        return None

    gains = wb_gains(rgb, paper)
    comp = largest_powder(powder)
    ys, xs = np.where(comp)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    side = max(48, int(min(bw, bh) * BASE_FRACTION))
    dist = ndi.distance_transform_edt(comp)

    crops, meta = [], []
    for k, off in enumerate(OFFSETS, start=1):
        px = int(cx + off * bw)
        found = best_square(comp, px, cy, side)
        if found is None:
            found = distance_square(comp, dist, px, cy, side)
        if found is None:  # keep three crops per source, even if one is a repeat
            found = distance_square(comp, dist, cx, cy, max(48, side // 2))
        x, y, s, frac = found
        patch = np.clip(rgb[y:y + s, x:x + s] * gains, 0, 255).astype(np.uint8)
        crops.append(Image.fromarray(patch).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS))
        meta.append({"crop": k, "x": x, "y": y, "size": s, "powder_frac": round(frac, 4)})
    return crops, meta


def main() -> None:
    times = capture_times()
    spreads = spread_ids(times)
    group_sizes = defaultdict(int)
    for g in spreads.values():
        group_sizes[g] += 1
    # Only use real spread bursts (>=10 frames). This drops notebook/label
    # photos and one-off setup shots that happen to pass the powder mask.
    allowed = {stem for stem, g in spreads.items() if group_sizes[g] >= 10}
    OUT.mkdir(exist_ok=True)

    manifest: list[dict] = []
    for cls_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        cls = cls_dir.name
        files = sorted(p for p in cls_dir.glob("*.jpg"))
        kept = 0
        for p in files:
            if kept >= PER_CLASS:
                break
            if p.stem not in allowed:
                continue
            made = make_crops(p)
            if made is None:
                continue
            crops, meta = made
            for c, m in zip(crops, meta):
                for k in (m["crop"],):
                    dst_dir = OUT / f"batch_{k}" / cls
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    name = f"{p.stem}_c{k}.jpg"
                    c.save(dst_dir / name, quality=92)
                    manifest.append({
                        "batch": f"batch_{k}",
                        "class": cls,
                        "file": name,
                        "source": p.name.replace(".jpg", ".HEIC"),
                        "spread": spreads.get(p.stem, ""),
                        "crop": k,
                        "x": m["x"], "y": m["y"], "size": m["size"],
                        "powder_frac": m["powder_frac"],
                    })
            kept += 1
        print(f"[{cls}] kept {kept} source images")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    print(f"[done] {len(manifest)} crops")
    for b in sorted({m['batch'] for m in manifest}):
        for cls in sorted({m['class'] for m in manifest}):
            n = sum(1 for m in manifest if m['batch'] == b and m['class'] == cls)
            print(f"  {b}/{cls}: {n}")


if __name__ == "__main__":
    main()
