# Dataset Card

## Object

**Laptop.**

**Justification:**
- **Availability** — readily on hand, allowing fast, repeated,
  varied-angle capture without needing to source or purchase a prop
  object.
- **Ease of annotation** — a laptop (closed, viewed from above/front) has
  a simple, mostly-rigid rectangular silhouette with clear, high-contrast
  edges against most backgrounds, making polygon/mask annotation fast and
  low-ambiguity compared to an object with soft or irregular boundaries.
- **Known reference dimensions** — actual physical dimensions
  **330.2 mm × 228.6 mm**, verified by direct measurement, giving an
  unambiguous ground truth for accuracy validation.

## Collection strategy

- Initial dataset: **81 images** of the laptop, captured with the same
  calibrated camera used for the calibration step, across varied angles,
  distances, and backgrounds.
- **Labelling tool:** Roboflow (single class: `laptop`), exported in COCO
  segmentation format.
- **Initial split (81 images):**

  | Split | Count |
  |-------|-------|
  | Train | 57    |
  | Val   | 15    |
  | Test  | 9     |

- **Test split replaced:** the original 9-image test split was swapped
  out for **12 newly captured images**, each containing **both the
  checkerboard and the laptop in the same frame**. This was a deliberate
  change to support Step 3's measurement-accuracy validation directly —
  every test-set image can be run through the full pipeline (undistort →
  detect checkerboard → compute pixels/mm → segment → measure) without
  needing a separately calibrated reference for that shot.

- **Active dataset composition (post-swap): 57 train / 15 val / 12 test =
  84 images.**

## Annotation format

Each split has its own COCO-format annotation file, exported from
Roboflow:

```
dataset/annotations.json         # full original export
dataset/train/annotations.json
dataset/val/annotations.json
dataset/test/annotations.json
```

Each file holds per-image polygon segmentation annotations for the single
`laptop` class. For training, these polygon annotations are rasterized
into binary PNG masks (one per image, matched by filename stem), which is
the format `models/train.py`'s `SegmentationDataset` consumes directly from
each split's `images/` and `masks/` folders.

## Class distribution

Single class: `laptop`. No class imbalance concern, since this is a
binary (object vs. background) segmentation task rather than a
multi-class problem.

## Known limitation: fixed-resolution export

Roboflow's export step resized every image (and its corresponding mask)
to a **fixed 640×640** square. Because the source laptop photos are not
square (native capture resolution 9248×6936, aspect ratio ≈1.33:1), this
resize does **not preserve aspect ratio** — it stretches/compresses one
axis relative to the other.

This is identified as the most likely root cause of the measurement bias
documented in `MEASUREMENT_REPORT.md`: the model was trained on
aspect-distorted masks, so its predicted mask shape inherits that
distortion, which shows up asymmetrically between the width and height
measurements at inference time. A future iteration should use
aspect-ratio-preserving resizing (e.g. letterbox/pad-to-square) instead of
a direct stretch, both during the Roboflow export and at inference time.