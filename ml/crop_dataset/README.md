# Powder-only crop dataset

1,800 JPEG crops built from the 600 usable photos in `../image dataset/`.

```
crop_dataset/
  batch_1/   600 images = 6 classes x 100   (centre section)
  batch_2/   600 images = 6 classes x 100   (left-of-centre section)
  batch_3/   600 images = 6 classes x 100   (right-of-centre section)
  manifest.csv
```

Class folders in every batch: `0_`, `10_`, `20_`, `30_`, `40_`, `50_`.
Each crop is 384 x 384 px, JPEG quality 92.

> **Note on the `30_` and `40_` folders.** The two batches were photographed
> the other way round: everything in `30_` is the 40% spread and everything in
> `40_` is the 30% spread. The folders were left as captured; the correction
> lives in `LABEL_MAP` in `../train_color_6class.py`, which relabels them when
> training. Do not re-sort the folders by hand.

## How they were made

1. White-balance the full frame against the paper around the spread
   (`wb_gains` from `prepare_crop.py`).
2. Segment the powder (saturation mask), keep the largest connected mass.
3. Cut three square sections from inside that mass: centre, left, right.
4. Require at least 97% of every crop to be powder; shrink/shift until it is.

Result: no paper, no tabletop, no hand, no notebook text in any crop.
Minimum powder coverage is 0.97; median is 1.00.

Only frames from real spread bursts (>= 10 photos in one capture) were used.
Notebook/label photos and one-off setup shots were excluded, which is why the
count is exactly 100 sources per class.

## Do not split by batch

The three batches are three views of the **same 600 photos**. `batch_1` is not
a training set and `batch_2` is not a validation set. Splitting that way leaks
the same physical spread into both sides.

Use the `spread` column in `manifest.csv` for leave-one-spread-out evaluation:

| Class | Independent spreads in this dataset |
|---|---:|
| 0% | 5 |
| 10% | 5 |
| 20% | 3 |
| 30% | 2 |
| 40% | 2 |
| 50% | 2 |

Also note that consecutive source photos are 1-2 seconds apart, so the 100
sources per class are still clusters of near-duplicates around those spreads.

## White balance at inference

The crops are already white-balanced. To classify a new photo the same way,
the paper must still be in the frame at capture time: white-balance against
it first, then crop the powder. A photo containing only powder with no paper
reference cannot be put on the same colour scale.

## manifest.csv

| Column | Meaning |
|---|---|
| `batch` | `batch_1`, `batch_2`, `batch_3` |
| `class` | class folder name |
| `file` | crop filename inside the batch/class folder |
| `source` | original HEIC filename |
| `spread` | capture-burst ID, e.g. `30_#0` (use for group-aware CV) |
| `crop` | 1 = centre, 2 = left, 3 = right |
| `x`, `y`, `size` | crop square in the 1600 px white-balanced frame |
| `powder_frac` | fraction of crop pixels inside the powder mask |

Rebuild with:

```bash
python make_crops.py
```
