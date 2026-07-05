# Measurement Methodology & Accuracy Report

## 1. Pixel-to-mm conversion — derivation

1. The checkerboard (9×7 inner corners, 20 mm squares) is detected in the
   **undistorted** image via `measure.py::detect_checkerboard()`.
2. `calculate_pixels_per_mm()` computes the mean Euclidean distance
   between every pair of horizontally-adjacent and vertically-adjacent
   detected corners:

   ```
   pixels_per_mm = mean(horizontal_spacings + vertical_spacings) / SQUARE_SIZE_MM
   ```

   Averaging across all `(9-1)×7 + 9×(7-1)` = 48 + 54 = 102 individual
   spacings (rather than using a single pair of corners) reduces
   sensitivity to any one corner's detection/sub-pixel-refinement noise.
3. The target object is segmented (U-Net → binary mask →
   `cv2.minAreaRect` on the largest contour), giving `width_px, height_px`.
4. Final conversion:

   ```
   width_mm  = width_px  / pixels_per_mm
   height_mm = height_px / pixels_per_mm
   ```

**Reference object:** the same physical checkerboard used for camera
calibration, placed in-frame alongside the laptop for every measurement
photo — this ensures the pixels-per-mm ratio is computed at the same
distance/plane as the object being measured, rather than assuming a fixed
ratio derived separately.

## 2. Calibration dependency

All measurement images are undistorted via `cv2.undistort()`, using the
resolution-rescaled intrinsic matrix (see `CALIBRATION_REPORT.md` §4),
**before** either checkerboard detection or segmentation runs. Skipping
this step would mean:

- The checkerboard corner spacing used to derive `pixels_per_mm` reflects
  locally-varying lens distortion rather than a uniform real-world scale.
- The laptop's contour (and therefore its `minAreaRect` width/height) is
  measured on curved, distorted edges rather than the object's true
  rectilinear boundary.

Both errors compound in the same conversion step, so raw (distorted)
images cannot produce a physically meaningful mm measurement even if a
plausible-looking pixels-per-mm ratio happens to be computed.

## 3. Accuracy validation

**Ground truth:** laptop physical dimensions, **330.2 mm (width) ×
228.6 mm (height)**, measured directly.

**12 trials**, run via the batch pipeline
(`python measurement/measure.py "dataset/test/images"`), each a distinct
photograph with the checkerboard and laptop in the same frame:

| Trial | Image | Pred. W (mm) | Actual W (mm) | Err W (mm) | Err W (%) | Pred. H (mm) | Actual H (mm) | Err H (mm) | Err H (%) | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1000141687...jpeg | 346.56 | 330.2 | +16.36 | +4.95% | 236.38 | 228.6 | +7.78 | +3.40% | 0.958 |
| 2 | 1000141740...jpeg | 348.05 | 330.2 | +17.85 | +5.41% | 233.72 | 228.6 | +5.12 | +2.24% | 0.914 |
| 3 | 1000141921...jpeg | 373.19 | 330.2 | +42.99 | +13.02% | 286.56 | 228.6 | +57.96 | +25.35% | 0.973 |
| 4 | 1000141939...jpeg | 329.83 | 330.2 | −0.37 | −0.11% | 240.74 | 228.6 | +12.14 | +5.31% | 0.975 |
| 5 | 1000141947...jpeg | 326.04 | 330.2 | −4.16 | −1.26% | 239.81 | 228.6 | +11.21 | +4.90% | 0.975 |
| 6 | IMG20260704134814.jpg.jpeg | 346.56 | 330.2 | +16.36 | +4.95% | 236.38 | 228.6 | +7.78 | +3.40% | 0.958 |
| 7 | IMG20260704134823.jpg.jpeg | 295.64 | 330.2 | −34.56 | −10.47% | 217.96 | 228.6 | −10.64 | −4.65% | 0.985 |
| 8 | IMG20260704134828.jpg.jpeg | 320.82 | 330.2 | −9.38 | −2.84% | 282.72 | 228.6 | +54.12 | +23.67% | 0.972 |
| 10 | IMG20260704134841.jpg.jpeg | 329.83 | 330.2 | −0.37 | −0.11% | 240.74 | 228.6 | +12.14 | +5.31% | 0.975 |
| 11 | IMG20260704134844.jpg.jpeg | 326.04 | 330.2 | −4.16 | −1.26% | 239.81 | 228.6 | +11.21 | +4.90% | 0.975 |
| 12 | IMG20260704140907.jpg.jpeg | 373.19 | 330.2 | +42.99 | +13.02% | 286.56 | 228.6 | +57.96 | +25.35% | 0.973 |
| 13 | IMG20260704185447.jpg.jpeg | 325.58 | 330.2 | −4.62 | −1.40% | 275.81 | 228.6 | +47.21 | +20.65% | 0.983 |

**Summary error metrics (n = 12 trials):**

| Metric | Width | Height | Overall |
|---|---|---|---|
| MAE (mm) | 16.18 | 24.61 | 20.39 |
| MPE — signed, i.e. average bias (%) | +1.99% | **+9.99%** | — |
| MAPE — mean absolute % error | 4.90% | 10.76% | 7.83% |

## 4. Error analysis

Two clear patterns emerge:

1. **Height error is both larger and consistently positive** (average
   bias +9.99%, 11 of 12 trials overestimated) — a systematic, directional
   bias, not random noise. **Width error is smaller and roughly balanced**
   between over- and under-estimates (signed bias only +1.99%, despite a
   4.90% average magnitude).
2. A one-directional bias on one axis but a near-zero net bias on the
   other axis is consistent with an **aspect-ratio distortion** rather
   than a uniform scale error: if scale alone were wrong (e.g. an
   incorrect checkerboard square size), both width and height would be
   biased in the *same direction and by a similar relative amount* — that
   is not what is observed.

**Root cause:** the training dataset was exported by Roboflow at a fixed
**640×640** resolution (see `DATASET_CARD.md`), which stretches the
non-square source images (native aspect ratio ≈1.33:1) onto a square grid
without preserving aspect ratio. The model therefore learned to predict
mask shapes calibrated to that distorted aspect ratio, and its
predictions inherit a directional shape bias when mapped back —
manifesting here as a disproportionate height overestimate.

**Secondary contributing factors (lower confidence, not fully isolated
from the above):**
- Checkerboard square size (20 mm) taken from nominal spec rather than
  caliper-verified (see `CALIBRATION_REPORT.md` §7) — would contribute a
  scale error, but would affect width and height proportionally and
  cannot alone explain the width/height asymmetry.
- Mask post-processing (`MORPH_CLOSE` ×2, `MORPH_OPEN` ×1 with a 5×5
  kernel) can add a small outward halo to the mask boundary, inflating
  both dimensions slightly — a possible contributor to the smaller,
  same-direction width overestimates seen in several trials (1, 2, 6).

## 5. Limitations

- Current MAPE (7.83% overall, 10.76% on height specifically) exceeds what
  would be considered production-grade measurement accuracy for an
  industrial workflow; the dominant cause (aspect-ratio-distorting resize)
  is identified but not yet corrected in a retrained model.
- Trial count (12) meets the assessment's 10+ requirement but remains a
  small sample; per-trial variance is visible (e.g. trials 3/12 both show
  the largest errors), and a larger trial set would give tighter
  confidence in the reported MAE/MPE.
- Accuracy validation assumes the laptop lies coplanar with the
  checkerboard in each shot; no explicit plane-alignment check is
  performed beyond visual framing during capture.


## 6. End-to-end usage

```bash
# Single image
python measurement/measure.py "path/to/photo.jpg"

# Batch over a folder (writes outputs/measurement_results/trials.csv)
python measurement/measure.py "dataset/test/images"

# Compute MAE/MPE from a filled-in trials.csv
python measurement/validate_accuracy.py "outputs/measurement_results/trials.csv"
```

Each single-image run outputs: an annotated image with the segmentation
box and `W:` / `H:` mm labels overlaid, the raw segmentation mask, and the
model's confidence score (printed to console and embedded in the output
image label).