"""Train and honestly evaluate the colour-based 5-class model.

Evaluation is leave-one-spread-out (LOSO), not a random image split:
every frame of a physical spread is held out together, so burst frames
cannot leak between train and test.

Run:
    python train_color.py

Writes:
    model/color_centroids.json   nearest-centroid model (G, B)
    model/color_features.csv     per-image features, for audit
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from color_model import powder_rgb
from prepare_crop import load_rgb

HERE = Path(__file__).resolve().parent
CLASS_ROOT = HERE / "turmeric_5class"
MODEL_OUT = HERE / "model" / "color_centroids.json"
MODEL_OUT_GB = HERE / "model" / "color_centroids_gb.json"
FEATURES_OUT = HERE / "model" / "color_features.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
GROUP_GAP_SECONDS = 6.0
SIZE = 640
CLASS_RE = re.compile(r"^(\d+)")

# Feature sets. `rgb` is the primary model; `gb` is the ablation that drops
# the exposure-sensitive red channel (R carries little starch signal).
FEATURE_SETS = {"rgb": [0, 1, 2], "gb": [1, 2]}


# --------------------------------------------------------------------- data
def find_class_dirs(root: Path) -> list[Path]:
    found = []
    for d in root.iterdir():
        if d.is_dir() and (m := CLASS_RE.match(d.name)):
            found.append((int(m.group(1)), d))
    return [d for _, d in sorted(found)]


def class_name(folder: Path) -> str:
    m = CLASS_RE.match(folder.name)
    if not m:
        raise ValueError(f"cannot parse class index from {folder.name!r}")
    return f"{int(m.group(1))}_percent"


def capture_time(path: Path) -> float:
    """EXIF DateTimeOriginal (with subseconds), else DateTime, else mtime."""
    try:
        exif = Image.open(path).getexif()
        ifd = exif.get_ifd(34665)
        stamp = ifd.get(36867) or exif.get(306)
        if stamp:
            sub = ifd.get(37520) or "0"
            try:
                frac = float("0." + str(sub).strip())
            except ValueError:
                frac = 0.0
            return datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S").timestamp() + frac
    except Exception:  # noqa: BLE001 - missing EXIF is not fatal
        pass
    return path.stat().st_mtime


def group_frames(times: list[float], gap: float = GROUP_GAP_SECONDS) -> list[int]:
    groups = [0] if times else []
    for prev, cur in zip(times, times[1:]):
        groups.append(groups[-1] + (1 if cur - prev > gap else 0))
    return groups


def load_rows(root: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[Path]]:
    feats, labels, groups, dates, names = [], [], [], [], []
    for ci, folder in enumerate(find_class_dirs(root)):
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            continue
        times = [capture_time(p) for p in files]
        grp = group_frames(times)
        print(f"[data] {folder.name:4s} {len(files):3d} images, {max(grp) + 1} spread(s)")
        for p, g, t in zip(files, grp, times):
            try:
                rgb = powder_rgb(load_rgb(p), SIZE)
            except ValueError as e:
                print(f"  skip {p.name}: {e}")
                continue
            feats.append(rgb)
            labels.append(ci)
            groups.append(f"{folder.name}#{g}")
            dates.append(datetime.fromtimestamp(t, tz=timezone.utc).isoformat())
            names.append(p)
    return np.asarray(feats), np.asarray(labels), groups, dates, names


# --------------------------------------------------------------------- model
def fit_centroids(x: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    centroids = np.stack([z[y == c].mean(axis=0) for c in range(n_classes)])
    return mean, std, centroids


def predict(x: np.ndarray, mean: np.ndarray, std: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = (x - mean) / std
    dist = np.linalg.norm(z[:, None, :] - centroids[None, :, :], axis=2)
    return dist.argmin(axis=1), dist


def logo_cv(x: np.ndarray, y: np.ndarray, groups: list[str], n_classes: int) -> dict:
    """Leave-one-spread-out CV. Returns image and spread level results."""
    y_true, y_pred = [], []
    spread_rows = []
    for held in sorted(set(groups)):
        test = np.array([g == held for g in groups])
        train = ~test
        if len(set(y[train])) < n_classes:
            print(f"  [warn] fold {held}: a class has no training spread")
            continue
        mean, std, cents = fit_centroids(x[train], y[train], n_classes)
        pred, dist = predict(x[test], mean, std, cents)
        y_true.extend(y[test].tolist())
        y_pred.extend(pred.tolist())
        spread_rows.append((held, int(y[test][0]), int(np.argmin(dist.mean(axis=0)))))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    st = np.asarray([r[1] for r in spread_rows])
    sp = np.asarray([r[2] for r in spread_rows])
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "image_accuracy": float((y_true == y_pred).mean()),
        "image_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "spread_rows": spread_rows,
        "spread_accuracy": float((st == sp).mean()),
    }


def summarize(name: str, res: dict, classes: list[str]) -> None:
    print(
        f"[LOSO] {name:14s} images {int((res['y_true'] == res['y_pred']).sum())}/{len(res['y_true'])}"
        f" = {res['image_accuracy'] * 100:5.1f}%"
        f" | spreads {int((np.asarray([r[1] for r in res['spread_rows']]) == np.asarray([r[2] for r in res['spread_rows']])).sum())}"
        f"/{len(res['spread_rows'])} = {res['spread_accuracy'] * 100:5.1f}%"
        f" | macro F1 {res['image_macro_f1']:.3f}"
    )
    print(confusion_matrix(res["y_true"], res["y_pred"]))
    print(classification_report(res["y_true"], res["y_pred"], target_names=classes, digits=3))


def main() -> None:
    root = CLASS_ROOT
    dirs = find_class_dirs(root)
    if not dirs:
        raise SystemExit(f"no class folders found under {root}")

    classes = [class_name(d) for d in dirs]
    pretty = {c: c.replace("_percent", "% impurity") for c in classes}
    if len(classes) != 5:
        raise SystemExit(f"expected 5 class folders, found {len(classes)}: {classes}")

    print(f"[root] {root}")
    rgb, y, groups, dates, names = load_rows(root)
    print(f"[data] {len(y)} images, {len(set(groups))} independent spreads\n")

    # Class/session overview: exposes confounds such as one class shot on a
    # different day.
    for ci, c in enumerate(classes):
        sel = y == ci
        day = sorted({d[:10] for d, m in zip(dates, sel) if m})
        print(f"  {c:12s} n={int(sel.sum()):3d} spreads={len({g for g, m in zip(groups, sel) if m}):2d} date(s)={day}")

    # Model feature set: rgb (primary) or gb (ablation).
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default="rgb",
                    help="rgb (primary) or gb (drop exposure-sensitive red)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    channels = FEATURE_SETS[args.features]
    feature_names = ["r", "g", "b"] if args.features == "rgb" else ["g", "b"]
    x_model = rgb[:, channels]

    print("\n--- leave-one-spread-out ---")
    res_rgb = logo_cv(rgb, y, groups, len(classes))
    summarize("all RGB", res_rgb, classes)
    res_gb = logo_cv(rgb[:, FEATURE_SETS["gb"]], y, groups, len(classes))
    summarize("G,B", res_gb, classes)
    res_model = res_rgb if args.features == "rgb" else res_gb
    summarize(f"{args.features.upper()} (model)", res_model, classes)

    print("\n[spreads] model prediction per held-out spread:")
    for held, true, pred in res_model["spread_rows"]:
        print(f"  {held:8s} true={classes[true]:12s} pred={classes[pred]}")

    # Final model on all data.
    mean, std, cents = fit_centroids(x_model, y, len(classes))
    z = (x_model - mean) / std
    own = np.linalg.norm(z - cents[y], axis=1)
    distance_scale = float(max(0.05, np.percentile(own, 95)))
    ood_threshold = float(max(0.6, 4.0 * distance_scale))

    model = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "features": args.features,
        "feature_names": feature_names,
        "channels": channels,
        "classes": classes,
        "pretty": pretty,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "centroids": [[float(v) for v in row] for row in cents],
        "distance_scale": distance_scale,
        "ood_threshold": ood_threshold,
        "n_images": int(len(y)),
        "n_spreads": len(set(groups)),
        "cv": {
            "protocol": "leave-one-spread-out",
            "rgb": {
                "image_accuracy": res_rgb["image_accuracy"],
                "image_macro_f1": res_rgb["image_macro_f1"],
                "spread_accuracy": res_rgb["spread_accuracy"],
            },
            "gb": {
                "image_accuracy": res_gb["image_accuracy"],
                "image_macro_f1": res_gb["image_macro_f1"],
                "spread_accuracy": res_gb["spread_accuracy"],
            },
        },
        "notes": (
            "Nearest centroid on mean white-balanced powder "
            + ", ".join(feature_names)
            + ". Confidence is an uncalibrated softmax over distance scaled by the "
            "typical within-class distance. 0_percent was captured in a single "
            "session; LOSO cannot test cross-session generalisation for that class."
        ),
    }

    out = args.out or (MODEL_OUT_GB if args.features == "gb" else MODEL_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
        fh.write("\n")

    with open(FEATURES_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class", "spread", "r", "g", "b"])
        for p, c, g, rgb_mean in zip(names, y, groups, rgb):
            w.writerow([p.name, classes[c], g, *[round(float(v), 2) for v in rgb_mean]])

    print(f"\n[save] {out}")
    print(f"[save] {FEATURES_OUT}")
    print(f"[model] features={','.join(feature_names)}  images={len(y)}  spreads={len(set(groups))}"
          f"  distance_scale={distance_scale:.3f}  ood_threshold={ood_threshold:.2f}")
    print("\n[note] 0% was captured on a different day/session from 10-50%.")
    print("[note] Report spread-level accuracy; image-level accuracy is optimistic.")


if __name__ == "__main__":
    main()
