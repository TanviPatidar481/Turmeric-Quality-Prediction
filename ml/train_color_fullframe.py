"""Train the shipped model on the exact feature the app computes.

The crop-trained model (`color_centroids_5class.json`) is built from 384 px
powder-only sections.  At serving time the app does something slightly
different: it white-balances the whole frame against the paper, masks the
powder, and takes the mean colour of that mask at 640 px.  Those two features
agree on only 96.7% of photos, and every disagreement sits on the fragile
0%/10% boundary.

This script closes that gap by construction: it computes features with
`color_model.powder_rgb`, the same function the API calls, so there is no
train/serve skew.  Evaluation is still leave-one-spread-out.

Run:
    python train_color_fullframe.py

Writes:
    model/color_centroids_5class_fullframe.json   <- deploy this one
    model/fullframe_color_features.csv
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
from sklearn.metrics import classification_report, confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from color_model import powder_rgb  # noqa: E402
from train_color_crops import (  # noqa: E402
    FOLDER_IMPURITY,
    SCHEMES,
    class_label,
    fit,
    loso,
    predict,
)
from prepare_crop import load_rgb  # noqa: E402

MANIFEST = HERE / "crop_dataset" / "manifest.csv"
CACHE = HERE / "tmp_newset_1600"          # optional 1600 px JPEG cache
ORIGINALS = HERE / "image dataset"
MODEL_DIR = HERE / "model"
FEATURES_OUT = MODEL_DIR / "fullframe_color_features.csv"


def sources() -> list[dict]:
    """One row per source photo, with its folder and capture-burst id."""
    seen: dict[str, dict] = {}
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        if r["source"] not in seen:
            seen[r["source"]] = {
                "source": r["source"],
                "folder": r["class"],
                "spread": r["spread"],
            }
    return [seen[k] for k in sorted(seen)]


def open_source(row: dict) -> Image.Image:
    stem = Path(row["source"]).stem
    cached = CACHE / row["folder"] / f"{stem}.jpg"
    if cached.is_file():
        return Image.open(cached).convert("RGB")
    return load_rgb(ORIGINALS / row["folder"] / row["source"])


def main() -> None:
    rows = sources()
    feats, keep, failed = [], [], []
    for i, row in enumerate(rows, 1):
        try:
            rgb = powder_rgb(open_source(row))
        except Exception as exc:  # noqa: BLE001  (missing paper/powder, corrupt)
            failed.append((row["source"], str(exc)))
            continue
        total = float(rgb.sum())
        if total <= 0:
            failed.append((row["source"], "zero colour"))
            continue
        feats.append([rgb[0] / total, rgb[1] / total])
        keep.append(row)

    print(f"[data] {len(feats)} photos usable, {len(failed)} rejected")
    for name, why in failed[:10]:
        print(f"  reject {name}: {why}")

    x = np.asarray(feats)
    folders = [r["folder"] for r in keep]
    spreads = np.asarray([r["spread"] for r in keep])
    truth = np.asarray([FOLDER_IMPURITY[f] for f in folders])

    print(f"[data] {len(set(spreads))} spreads, "
          f"{len(set(folders))} folder groups")

    summary = {}
    for scheme in ("5class", "6class"):
        cfg = SCHEMES[scheme]
        y = np.asarray([class_label(cfg["merge"].get(int(t), int(t))) for t in truth])
        classes = [class_label(v) for v in cfg["classes"]]

        good = [c for c in classes if (y == c).sum() >= 2]
        if len(good) < len(classes):
            print(f"[skip] {scheme}: missing classes {set(classes) - set(good)}")
            continue

        print(f"\n{'=' * 74}\n[{scheme}] {len(y)} photos")
        res = loso(x, y, spreads, classes, f"{scheme} full-frame", verbose=(scheme == "5class"))
        summary[scheme] = res

        if scheme != "5class":
            continue

        mean, std, cents = fit(x, y, classes)
        z = (x - mean) / std
        own = np.linalg.norm(z - cents[[classes.index(c) for c in y]], axis=1)
        scale = float(max(0.05, np.percentile(own, 95)))

        per_class = {
            c: {
                "n": int((y == c).sum()),
                "recall": float((res["y_pred"][res["y_true"] == c] == c).mean()),
                "spreads": len({s for s, m in zip(spreads, y == c) if m}),
            }
            for c in classes
        }

        model = {
            "version": 1,
            "created": datetime.now(timezone.utc).isoformat(),
            "task": "5-class turmeric impurity, trained on the serving feature",
            "scheme": scheme,
            "features": "chromaticity_rg",
            "feature_pipeline": (
                "powder_rgb(img, long_side=640): white-balance against the paper "
                "in frame, mask the powder, mean RGB of the mask; then "
                "r/(r+g+b), g/(r+g+b). This is byte-for-byte the function the "
                "API calls, so there is no train/serve skew."
            ),
            "feature_names": ["r_chroma", "g_chroma"],
            "classes": classes,
            "pretty": {c: c.replace("_percent", "% impurity") for c in classes},
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "centroids": [[float(v) for v in row] for row in cents],
            "distance_scale": scale,
            "ood_threshold": float(max(0.6, 4.0 * scale)),
            "n_photos": int(len(y)),
            "n_spreads": int(len(set(spreads))),
            "per_class": per_class,
            "confusion": {
                "labels": classes,
                "matrix": confusion_matrix(
                    res["y_true"], res["y_pred"], labels=classes
                ).tolist(),
            },
            "cv": {
                "protocol": "leave-one-spread-out",
                "photo_accuracy": res["crop_accuracy"],
                "photo_macro_f1": res["crop_macro_f1"],
                "spread_accuracy": res["spread_accuracy"],
            },
            "labels": {
                "folder_impurity": {k: class_label(v) for k, v in FOLDER_IMPURITY.items()},
                "merged": {class_label(k): class_label(v) for k, v in cfg["merge"].items()},
                "note": (
                    "Folders 30_ and 40_ were photographed the wrong way round; "
                    "FOLDER_IMPURITY corrects them. 30% is folded into 40% because "
                    "the two batches are only ~0.0024 apart in g-chromaticity "
                    "against 0.008-0.013 for every other neighbouring pair."
                ),
            },
            "notes": (
                "Trained on 600 photos, 19 independent spreads. Accuracy is at the "
                "photo level, not the crop level: each row is one physical photo."
            ),
        }
        out = MODEL_DIR / f"color_centroids_{scheme}_fullframe.json"
        out.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
        print(f"\n[save] {out}  ({out.stat().st_size / 1024:.1f} KB)")

    for scheme, res in summary.items():
        print(f"[{scheme}] photos {res['crop_accuracy'] * 100:5.1f}%  "
              f"spreads {res['spread_accuracy'] * 100:5.1f}%  "
              f"macro F1 {res['crop_macro_f1']:.3f}")

    y5 = np.asarray([class_label(SCHEMES["5class"]["merge"].get(int(t), int(t))) for t in truth])
    y6 = np.asarray([class_label(SCHEMES["6class"]["merge"].get(int(t), int(t))) for t in truth])
    with open(FEATURES_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "folder", "spread", "r_chroma", "g_chroma",
                    "class_5class", "class_6class"])
        for i, row in enumerate(keep):
            w.writerow([row["source"], row["folder"], row["spread"],
                        round(float(x[i, 0]), 5), round(float(x[i, 1]), 5),
                        y5[i], y6[i]])
    print(f"[save] {FEATURES_OUT}")


if __name__ == "__main__":
    main()
