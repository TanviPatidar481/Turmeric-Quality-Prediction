"""Train the colour model on the powder-only crop dataset.

Two label schemes are supported:

    5class (shipped)   0, 10, 20, 40, 50     -- 30% folded into 40%
    6class (reference) 0, 10, 20, 30, 40, 50

Folder correction
-----------------
The folders named ``30_`` and ``40_`` hold the batches the other way round:
the ``30_`` photos are the 40% spread and the ``40_`` photos are the 30% spread.
``FOLDER_IMPURITY`` below is the single place that records this.  In the
5-class scheme both folders fold into 40%, so the swap is absorbed with no
separate correction needed.

Evidence for the swap: before it, the class order in g-chromaticity ran
20 < 40 < 30 < 50 (a negative step), and back-solving the 20-to-50 trend put
the ``30_`` powder near 45% starch and the ``40_`` powder near 33%.

Evaluation
----------
Leave-one-spread-out, using the ``spread`` column in ``crop_dataset/manifest.csv``.
All three crops of a photo, and every photo in the same capture burst, stay on
the same side of the split.

Run:
    python train_color_crops.py

Writes:
    model/color_centroids_5class.json
    model/color_centroids_6class.json
    model/crop_color_features.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE / "crop_dataset"
MODEL_DIR = HERE / "model"
FEATURES_OUT = MODEL_DIR / "crop_color_features.csv"

# Folder name -> true impurity percent, after correcting the swapped batches.
FOLDER_IMPURITY = {
    "0_": 0,
    "10_": 10,
    "20_": 20,
    "30_": 40,
    "40_": 30,
    "50_": 50,
}

SCHEMES = {
    "6class": {"classes": [0, 10, 20, 30, 40, 50], "merge": {}},
    "5class": {"classes": [0, 10, 20, 40, 50], "merge": {30: 40}},
}


def chromaticity(x: np.ndarray) -> np.ndarray:
    """Illumination-robust powder colour: r/(r+g+b), g/(r+g+b)."""
    return x[:, :2] / np.maximum(x.sum(1, keepdims=True), 1.0)


def class_label(impurity: int) -> str:
    return f"{impurity}_percent"


def crop_mean(path: Path) -> np.ndarray:
    """Mean colour of the powder-like pixels in one already-balanced crop."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)
    powder = (sat > 0.50) & (a[..., 0] > 35)
    if powder.sum() < 0.5 * powder.size:
        powder = np.ones(powder.shape, dtype=bool)
    return a[powder].mean(0)


def load_rows() -> tuple[np.ndarray, list[str], list[str], list[str], list[str]]:
    manifest = list(csv.DictReader(open(ROOT / "manifest.csv", encoding="utf-8")))
    feats, folders, spreads, sources, files = [], [], [], [], []
    for r in manifest:
        path = ROOT / r["batch"] / r["class"] / r["file"]
        try:
            feats.append(crop_mean(path))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {path}: {exc}")
            continue
        folders.append(r["class"])
        spreads.append(r["spread"])
        sources.append(r["source"])
        files.append(f"{r['batch']}/{r['class']}/{r['file']}")
    return (np.asarray(feats), folders, spreads, sources, files)


def labels_for(folders: list[str], merge: dict[int, int]) -> np.ndarray:
    """Map each folder to its (possibly merged) class label."""
    truth = np.asarray([FOLDER_IMPURITY[f] for f in folders])
    merged = np.asarray([merge.get(int(t), int(t)) for t in truth])
    return np.asarray([class_label(v) for v in merged])


def fit(x: np.ndarray, y: np.ndarray, classes: list[str]):
    mean = x.mean(0)
    std = x.std(0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    cents = np.stack([z[y == c].mean(0) for c in classes])
    return mean, std, cents


def predict(x, mean, std, cents):
    z = (x - mean) / std
    d = np.linalg.norm(z[:, None, :] - cents[None, :, :], axis=2)
    return d.argmin(1), d


def group_means(x, y, keys):
    order: list[str] = []
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        if k not in buckets:
            order.append(k)
        buckets[k].append(i)
    gx = np.stack([x[buckets[k]].mean(0) for k in order])
    gy = np.asarray([y[buckets[k][0]] for k in order])
    return gx, gy, order


def loso(x, y, groups, classes, name, verbose=False):
    yt, yp, spread_rows = [], [], []
    for held in sorted(set(groups)):
        te = np.asarray([g == held for g in groups])
        tr = ~te
        mean, std, cents = fit(x[tr], y[tr], classes)
        p, d = predict(x[te], mean, std, cents)
        yt.extend(y[te])
        yp.extend([classes[i] for i in p])
        spread_rows.append((held, y[te][0], classes[int(np.argmin(d.mean(0)))], int(te.sum())))
    yt, yp = np.asarray(yt), np.asarray(yp)
    acc = float((yt == yp).mean())
    f1 = float(f1_score(yt, yp, average="macro"))
    st = np.asarray([r[1] for r in spread_rows])
    sp = np.asarray([r[2] for r in spread_rows])
    spread_acc = float((st == sp).mean())

    print(f"\n[LOSO] {name:22s} rows {int((yt == yp).sum()):4d}/{len(yt)} = {acc * 100:5.1f}%"
          f" | spreads {int((st == sp).sum()):2d}/{len(st)} = {spread_acc * 100:5.1f}%"
          f" | macro F1 {f1:.3f}")
    if verbose:
        print(confusion_matrix(yt, yp, labels=classes))
        print(classification_report(yt, yp, labels=classes, digits=3, zero_division=0))
        for held, t, p, n in spread_rows:
            print(f"  {held:8s} n={n:3d} true={t:12s} pred={p:12s} {'OK' if t == p else 'ERR'}")
    else:
        recall = " ".join(f"{c.split('_')[0]}%:{100 * (yp[yt == c] == c).mean():.0f}" for c in classes)
        print(f"         recall  {recall}")

    return {
        "crop_accuracy": acc,
        "crop_macro_f1": f1,
        "spread_accuracy": spread_acc,
        "spread_rows": spread_rows,
        "y_true": yt,
        "y_pred": yp,
    }


def train_scheme(scheme: str, x, folders, spreads, sources) -> dict:
    cfg = SCHEMES[scheme]
    classes = [class_label(v) for v in cfg["classes"]]
    y = labels_for(folders, cfg["merge"])

    print(f"\n{'=' * 78}\n[{scheme}] {len(y)} crops, {len(set(sources))} photos, "
          f"{len(set(spreads))} spreads, {len(classes)} classes")

    # Sanity: more starch must mean a higher g-chromaticity.
    chrom = chromaticity(x)
    print("[sanity] mean g-chromaticity by class")
    g = {c: chrom[y == c, 1].mean() for c in classes}
    ordered = [g[c] for c in classes]
    for i, c in enumerate(classes):
        step = "" if i == 0 else f"  step {ordered[i] - ordered[i - 1]:+.4f}"
        print(f"  {c:12s} {ordered[i]:.4f}{step}")
    inversions = [(classes[i], classes[i + 1]) for i in range(len(classes) - 1)
                  if ordered[i + 1] < ordered[i]]
    print(f"  {'WARNING: non-monotonic at ' + str(inversions) if inversions else 'monotonic increasing: OK'}")

    results = {
        "rgb": loso(x[:, :3], y, spreads, classes, "R,G,B"),
        "gb": loso(x[:, 1:3], y, spreads, classes, "G,B"),
        "chromaticity": loso(chrom, y, spreads, classes, "r,g chromaticity", verbose=True),
    }

    gc, gy, gkeys = group_means(chrom, y, sources)
    gspread = dict(zip(sources, spreads))
    results["source_means_chromaticity"] = loso(
        gc, gy, [gspread[k] for k in gkeys], classes, "source means (chrom)")

    # Final model: chromaticity, trained on every crop.
    mean, std, cents = fit(chrom, y, classes)
    z = (chrom - mean) / std
    own = np.linalg.norm(z - cents[[classes.index(c) for c in y]], axis=1)
    scale = float(max(0.05, np.percentile(own, 95)))
    ood = float(max(0.6, 4.0 * scale))

    model = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "task": f"{len(classes)}-class turmeric impurity from powder-only crops",
        "scheme": scheme,
        "features": "chromaticity_rg",
        "feature_names": ["r_chroma", "g_chroma"],
        "feature_note": (
            "r/(r+g+b) and g/(r+g+b) of the mean RGB of saturated powder pixels "
            "in an already white-balanced powder-only crop"
        ),
        "classes": classes,
        "pretty": {c: c.replace("_percent", "% impurity") for c in classes},
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "centroids": [[float(v) for v in row] for row in cents],
        "distance_scale": scale,
        "ood_threshold": ood,
        "n_crops": int(len(y)),
        "n_sources": len(set(sources)),
        "n_spreads": len(set(spreads)),
        "cv": {
            "protocol": "leave-one-spread-out",
            **{k: {m: results[k][m] for m in
                   ("crop_accuracy", "crop_macro_f1", "spread_accuracy")}
               for k in results},
        },
        "labels": {
            "folder_impurity": {k: class_label(v) for k, v in FOLDER_IMPURITY.items()},
            "merged": {class_label(k): class_label(v) for k, v in cfg["merge"].items()},
            "reason": (
                "Folders 30_ and 40_ were photographed the wrong way round; "
                "FOLDER_IMPURITY corrects them. In the 5-class scheme 30% is "
                "folded into 40% because the two batches are only ~0.0024 apart "
                "in g-chromaticity, versus 0.008-0.013 for every other "
                "neighbouring pair, so they cannot be separated honestly."
            ),
        },
        "notes": (
            "Expects a powder-only, white-balanced crop. Chromaticity removes the "
            "camera's colour response, which is why raw RGB and G,B score far "
            "worse. LOSO groups every crop of a capture burst. Spreads per class: "
            "5,5,3,4,2 for the 5-class scheme."
        ),
    }
    out = MODEL_DIR / f"color_centroids_{scheme}.json"
    out.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"\n[save] {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print(f"[model] {scheme}  crops={len(y)}  sources={model['n_sources']}  "
          f"spreads={model['n_spreads']}  scale={scale:.4f}  ood={ood:.3f}")
    return model


def main() -> None:
    x, folders, spreads, sources, files = load_rows()
    print(f"[data] {len(x)} crops, {len(set(sources))} source photos, "
          f"{len(set(spreads))} spreads")

    for scheme in ("5class", "6class"):
        train_scheme(scheme, x, folders, spreads, sources)

    chrom = chromaticity(x)
    y5 = labels_for(folders, SCHEMES["5class"]["merge"])
    y6 = labels_for(folders, SCHEMES["6class"]["merge"])
    with open(FEATURES_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "folder", "spread", "source", "r", "g", "b",
                    "r_chroma", "g_chroma", "class_5class", "class_6class"])
        for i, f in enumerate(files):
            w.writerow([f, folders[i], spreads[i], sources[i],
                        *[round(float(v), 2) for v in x[i]],
                        round(float(chrom[i, 0]), 5), round(float(chrom[i, 1]), 5),
                        y5[i], y6[i]])
    print(f"\n[save] {FEATURES_OUT}")


if __name__ == "__main__":
    main()
