"""
Crop turmeric-spread photos down to the powder, and flag frames that are
near-duplicates of each other.

Why this exists
---------------
The camera frames the whole sheet of paper; the model only cares about the
powder. Cropping in code gives every image the same geometry, which is what
makes colour comparisons between classes meaningful.

Two rules baked in:
  * The crop is a SQUARE fitted to the powder mass, so resizing to 224x224
    never stretches or squashes the texture.
  * Cropping happens AFTER an optional white-balance correction that uses the
    paper still visible in the frame. Once you crop to powder only, the
    reference is gone - so correct first, crop second.

Usage
-----
    python prepare_crop.py ../pure --out ../data/crops/0_percent
    python prepare_crop.py ../pure --out ../data/crops/0_percent --dedupe
    python prepare_crop.py ../pure --out out --size 640 --wb --report report.csv

The source folder is never modified.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff"}


# --------------------------------------------------------------------------- io
def load_rgb(path: Path) -> Image.Image:
    """Open an image as RGB, converting HEIC through macOS `sips` if needed."""
    if path.suffix.lower() not in {".heic", ".heif"}:
        return Image.open(path).convert("RGB")

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "converted.png"
        subprocess.run(
            [
                "sips",
                "-s", "format", "png",
                "--matchTo", "/System/Library/ColorSync/Profiles/sRGB Profile.icc",
                str(path),
                "--out", str(png),
            ],
            check=True,
            capture_output=True,
        )
        return Image.open(png).convert("RGB")


# ------------------------------------------------------------------- detection
def masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the frame into 'powder' and 'paper' pixels.

    Turmeric is strongly yellow: high saturation. Paper is bright and
    near-neutral. Shadowed paper can drift warm, so the paper test uses a
    looser saturation bound than the powder test.
    """
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)

    paper = (mx > 140) & (sat < 0.16)
    powder = (sat > 0.50) & (rgb[..., 0] > 35) & ~paper

    powder = ndi.binary_opening(powder, iterations=2)
    powder = ndi.binary_closing(powder, iterations=3)
    powder = _largest_component(powder)
    powder = ndi.binary_fill_holes(powder)
    return powder, paper


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


# ----------------------------------------------------------------------- crop
def square_crop(
    rgb: np.ndarray,
    powder: np.ndarray,
    min_powder_fraction: float = 1.0,
    margin_fraction: float = 0.02,
) -> tuple[int, int, int]:
    """Largest centred square whose area is at least `min_powder_fraction` powder.

    Pale pixels (bright and weakly saturated - paper slivers, dust, blown
    highlights) are excluded and padded out before fitting. They sit *inside*
    the powder mask at the ragged spread edge, so eroding the mask alone does
    not remove them; without this step a white sliver survives in the corner of
    the crop and reads to the model as starch. The crop is then eroded by a
    safety margin so it never touches the boundary.

    Starts from the usable mask's bounding box, centred on its centroid, and
    shrinks until the powder covers the required fraction. Shrinking instead of
    nudging keeps the crop centred on the mass, which is what makes the geometry
    repeatable between samples.
    """
    if not powder.any():
        raise ValueError("no powder detected")

    h, w = powder.shape
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)

    pale = (mx > 185) & (sat < 0.35)
    pad = max(4, int(round(min(h, w) * 0.008)))
    pale = ndi.distance_transform_edt(~pale) <= pad

    usable = powder & ~pale
    radius = max(3, int(round(min(h, w) * margin_fraction)))
    usable = ndi.distance_transform_edt(usable) > radius
    if usable.sum() < 1000:
        raise ValueError("powder mass too small after excluding pale pixels")

    cy, cx = ndi.center_of_mass(usable)
    ys, xs = np.nonzero(usable)
    side = int(min(usable.sum() ** 0.5, np.ptp(ys), np.ptp(xs)))
    side = max(side, 32)
    while side > 32:
        half = side // 2
        y0 = int(round(cy)) - half
        x0 = int(round(cx)) - half
        y0 = min(max(y0, 0), h - side)
        x0 = min(max(x0, 0), w - side)
        window = usable[y0:y0 + side, x0:x0 + side]
        ok = window.all() if min_powder_fraction >= 1.0 else window.mean() >= min_powder_fraction
        if ok:
            return x0, y0, side
        side = int(side * 0.96)

    raise ValueError("could not fit a clean square inside the powder")


# -------------------------------------------------------------------- quality
def sharpness(gray: np.ndarray) -> float:
    lap = (
        np.roll(gray, 1, 0)
        + np.roll(gray, -1, 0)
        + np.roll(gray, 1, 1)
        + np.roll(gray, -1, 1)
        - 4 * gray
    )[1:-1, 1:-1]
    return float(lap.var())


def wb_gains(rgb: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Per-channel gains that make the brightest paper neutral."""
    if paper.sum() < 500:
        return np.ones(3)
    px = rgb[paper]
    lum = px.mean(axis=1)
    bright = px[lum > np.percentile(lum, 75)]
    ref = bright.mean(axis=0)
    return ref.mean() / np.maximum(ref, 1e-6)


# ---------------------------------------------------------------- dedupe group
def capture_time(path: Path) -> float:
    """EXIF capture time, falling back to the file's modification time."""
    try:
        exif = Image.open(path).getexif()
        stamp = exif.get(36867) or exif.get(306)  # DateTimeOriginal, DateTime
        if stamp:
            return datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:  # noqa: BLE001 - missing or unparsable EXIF is not fatal
        pass
    return path.stat().st_mtime


def group_frames(times: list[float], gap_seconds: float) -> list[int]:
    """Assign each frame a group id, splitting whenever the clock jumps.

    One group means one physical spread. Re-levelling the powder takes longer
    than `gap_seconds`, so the next spread lands in the next group - while the
    three or four frames you rattle off of a single spread stay together.

    Counting frames of one spread as independent examples is how a test split
    ends up with the same powder on both sides of it.
    """
    groups = [0] if times else []
    for prev, cur in zip(times, times[1:]):
        groups.append(groups[-1] + (1 if cur - prev > gap_seconds else 0))
    return groups


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="folder of original photos")
    ap.add_argument("--out", type=Path, required=True, help="where crops are written")
    ap.add_argument("--size", type=int, default=512, help="output side in pixels")
    ap.add_argument("--min-powder", type=float, default=1.0,
                    help="fraction of the crop that must be powder")
    ap.add_argument("--wb", action="store_true",
                    help="white-balance using the paper visible in the frame")
    ap.add_argument("--dedupe", action="store_true",
                    help="keep only the sharpest frame per capture group")
    ap.add_argument("--group-gap", type=float, default=6.0,
                    help="seconds between frames that starts a new sample group "
                         "(re-levelling takes longer than this)")
    ap.add_argument("--report", type=Path, default=None, help="optional CSV path")
    ap.add_argument("--prune", action="store_true",
                    help="delete crops in --out left over from an earlier run")
    args = ap.parse_args()

    files = [
        p for p in args.src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise SystemExit(f"no images found in {args.src}")
    times = {p: capture_time(p) for p in files}
    files.sort(key=lambda p: (times[p], p.name))

    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in files:
        try:
            img = load_rgb(path)
            rgb = np.asarray(img).astype(np.float32)

            # Detect on a downscaled copy, crop from the full-resolution frame:
            # mask work is the slow part and the detection does not need 12 MP.
            step = max(1, round(max(img.size) / 1024))
            small = img.resize(
                (img.width // step, img.height // step), Image.BILINEAR
            )
            small_rgb = np.asarray(small).astype(np.float32)

            powder, paper = masks(small_rgb)
            sx, sy, ss = square_crop(small_rgb, powder, args.min_powder)
            x0, y0, side = sx * step, sy * step, ss * step

            gains = wb_gains(small_rgb, paper) if args.wb else np.ones(3)
            corrected = np.clip(rgb * gains, 0, 255)

            crop = Image.fromarray(corrected[y0:y0 + side, x0:x0 + side].astype(np.uint8))
            crop = crop.resize((args.size, args.size), Image.LANCZOS)

            c = np.asarray(crop).astype(np.float32).reshape(-1, 3)

            rows.append(
                {
                    "path": path,
                    "crop": crop,
                    "time": times[path],
                    "x0": x0, "y0": y0, "side": side,
                    "powder_pct": float(powder.mean() * 100),
                    "crop_powder": float((powder[sy:sy + ss, sx:sx + ss]).mean()),
                    "mean_rgb": c.mean(axis=0),
                    "blue_median": float(np.median(c[:, 2])),
                    "sharpness": sharpness(
                        np.asarray(img.convert("L").resize((512, 384), Image.LANCZOS))
                        .astype(np.float32)
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep going, report the failure
            print(f"  skip {path.name}: {exc}")

    if not rows:
        raise SystemExit("nothing could be processed")

    groups = group_frames([r["time"] for r in rows], args.group_gap)

    keep: list[int] = []
    if args.dedupe:
        best: dict[int, int] = {}
        for i, g in enumerate(groups):
            if g not in best or rows[i]["sharpness"] > rows[best[g]]["sharpness"]:
                best[g] = i
        keep = sorted(best.values())
    else:
        keep = list(range(len(rows)))

    print(f"\n{'file':16s} {'crop':>14s} {'pow%':>6s} {'RGB':>18s} {'B med':>6s} {'grp':>4s}")
    for i, r in enumerate(rows):
        mark = "" if i in keep else "  (dup)"
        print(
            f"{r['path'].name:16s} {r['side']:5d}px@{r['x0']:4d},{r['y0']:4d} "
            f"{r['powder_pct']:5.1f}% "
            f"({r['mean_rgb'][0]:4.0f},{r['mean_rgb'][1]:4.0f},{r['mean_rgb'][2]:3.0f}) "
            f"{r['blue_median']:6.0f} {groups[i]:4d}{mark}"
        )

    planned = {f"{rows[i]['path'].stem}_g{groups[i]:02d}.png" for i in keep}
    stale = sorted(
        p for p in args.out.glob("*_g[0-9][0-9].png") if p.name not in planned
    )
    if stale and args.prune:
        for p in stale:
            p.unlink()
        print(f"\npruned {len(stale)} stale crop(s) from a previous run")
    elif stale:
        print(
            f"\nwarning: {len(stale)} crop(s) in {args.out} are left over from an earlier\n"
            "  run and no longer match this one. Re-run with --prune to remove them,\n"
            "  otherwise the folder mixes two versions of the dataset."
        )

    for i in keep:
        r = rows[i]
        name = f"{r['path'].stem}_g{groups[i]:02d}.png"
        r["crop"].save(args.out / name)

    n_groups = len(set(groups))
    print(
        f"\n{len(rows)} frames -> {n_groups} independent sample group(s); "
        f"{len(keep)} file(s) written to {args.out}"
    )
    if n_groups < len(rows):
        print(
            "  Frames sharing a group are the same physical spread. Keep them in the\n"
            "  same train/val/test split, or run with --dedupe to keep one per group."
        )

    means = np.array([rows[i]["mean_rgb"] for i in keep])
    print(
        "  kept-set mean RGB (%.1f, %.1f, %.1f)  std (%.2f, %.2f, %.2f)"
        % (*means.mean(axis=0), *means.std(axis=0))
    )
    print(
        "  blue median range %.0f-%.0f  |  black/blue floor (<5) on %.0f%% of pixels"
        % (
            min(rows[i]["blue_median"] for i in keep),
            max(rows[i]["blue_median"] for i in keep),
            100 * np.mean([rows[i]["blue_median"] < 5 for i in keep]),
        )
    )

    if args.report:
        with open(args.report, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "group", "kept", "x0", "y0", "side",
                        "powder_pct", "mean_r", "mean_g", "mean_b",
                        "blue_median", "sharpness"])
            for i, r in enumerate(rows):
                w.writerow([
                    r["path"].name, groups[i], int(i in keep), r["x0"], r["y0"],
                    r["side"], round(r["powder_pct"], 2),
                    *[round(v, 2) for v in r["mean_rgb"]],
                    round(r["blue_median"], 1), round(r["sharpness"], 1),
                ])
        print(f"  report written to {args.report}")


if __name__ == "__main__":
    main()
