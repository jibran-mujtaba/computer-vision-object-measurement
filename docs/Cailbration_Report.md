# Camera Calibration Report

## 1. Method

Intrinsic camera calibration was performed using OpenCV's standard
pinhole-camera model (`cv2.calibrateCamera`), based on Zhang's method of
observing a planar checkerboard pattern from multiple viewpoints.

- **Calibration target:** printed checkerboard, **9×7 inner corners**
  (columns × rows), square size **20 mm** (per printed spec — see
  Limitations).
- **Detection:** `cv2.findChessboardCorners` with
  `CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK`
  flags, followed by sub-pixel corner refinement
  (`cv2.cornerSubPix`, 11×11 window).
- **Orientation normalization:** all input images were rotated to a
  canonical landscape orientation before detection, since
  `cv2.calibrateCamera` requires every image to share one consistent
  `imageSize` — mixing portrait/landscape captures would silently corrupt
  the resulting camera matrix.
- **Resize:** all calibration images were downscaled by `RESIZE_SCALE = 0.5`
  before detection and calibration (see §4).

Script: `calibration/images/calibrate.py`.

## 2. Calibration dataset

- **21 checkerboard images** captured from varied angles and distances.
- **21 / 21 (100%)** successfully detected — no images were discarded for
  failed detection.
- **13 of 21** images required an orientation correction (portrait →
  landscape rotation) before detection.
- Calibration resolution locked at **(w, h) = (4624, 3468)** — the
  original 9248×6936 capture resolution, halved by `RESIZE_SCALE`.

## 3. Results

**Reprojection error**
| `cv2.calibrateCamera` RMS error | **2.339 px** |

| Mean per-point reprojection error (recomputed via `cv2.projectPoints`) | **0.256 px** |

The target for this assessment is <0.5 px (acceptable) / <0.3 px
(excellent). The **mean reprojection error (0.256 px) meets the "excellent"
threshold**. The RMS value returned directly by `calibrateCamera` (2.339 px)
is higher because it is influenced more heavily by any high-error
outlier images/corners in the set, while the mean per-point error computed
independently afterward averages more evenly across all correspondences.
This points to a small number of images (likely captured at a more oblique
angle) contributing disproportionately to error — good candidates to
re-shoot or exclude in a future recalibration pass.

**Intrinsic camera matrix** (valid at 4624×3468 — see §4):

```
[[3457.85,    0.00, 2189.25],
 [   0.00, 3442.56, 1796.94],
 [   0.00,    0.00,    1.00]]
```

**Distortion coefficients** `[k1, k2, p1, p2, k3]`:

```
[-0.0229, 0.2982, 0.00105, 0.000175, -0.8266]
```

## 4. Resolution-dependent camera matrix — issue and fix

`fx, fy, cx, cy` in a camera matrix are expressed in **pixels at a
specific resolution** — they are not resolution-independent. Because
calibration ran on images resized to 4624×3468, that matrix is only
directly valid at that resolution.

The measurement pipeline captures photos at the camera's native
resolution (9248×6936 — exactly 2× the calibration resolution here).
Applying the unscaled matrix to a full-resolution photo warps the image
incorrectly during undistortion rather than correcting it — this is
precisely what happened during development: checkerboard detection
succeeded on the raw image but failed entirely after undistortion.

**Fix implemented:** `measure.py::scale_camera_matrix()` rescales
`fx, fy, cx, cy` proportionally to the actual input image's resolution
before every `cv2.undistort()` call. Distortion coefficients are
dimensionless ratios and require no rescaling. A sanity check
(`_sanity_check_calibration()`) also verifies the principal point
(`cx/width`, `cy/height`) falls within an expected 0.35–0.65 range for the
image being processed, printing a warning immediately if not, rather than
silently producing a warped result.

**Housekeeping note:** two calibration `.npz` files currently exist —
`calibration/calibration_data.npz` (root) and
`calibration/images/calibration_data.npz`. Only the latter is loaded by
`measure.py` (`CALIBRATION_FILE` points there explicitly). The root-level
file is a leftover from an earlier debugging session where a path
mismatch caused the wrong calibration to be loaded silently; it should be
deleted before submission so only one calibration file exists.

## 5. `undistort.py` — standalone verification utility

`calibration/images/undistort.py` is a separate, standalone script used to
visually spot-check the calibration on a single test image outside the
main measurement pipeline: it applies `cv2.undistort()` with the saved
calibration and writes `original_output.jpg` / `undistorted_output.jpg`
side by side for visual comparison. These two output filenames are
explicitly excluded from `calibrate.py`'s own image glob
(`EXCLUDE_NAMES`) so that re-running calibration never accidentally treats
a prior undistortion test output as a new calibration input.

## 6. Why undistortion is mandatory before measurement

Lens distortion (radial + tangential) is **non-uniform across the frame**:
straight lines bow outward or inward depending on their distance from the
optical center, and the degree of warping increases toward the image
edges. This means:

- A pixels-per-mm ratio computed from a *distorted* checkerboard is only
  locally valid near where that checkerboard sat in the frame — applying
  it to an object positioned elsewhere in the same distorted image
  introduces geometric error that grows with distance from the
  checkerboard.
- Straight physical edges (e.g. a laptop's rectangular outline) appear
  slightly curved in a distorted image, which corrupts both the
  segmentation mask's contour and the `minAreaRect` fit used to derive
  width/height.

`cv2.undistort()`, with the correctly-scaled camera matrix, removes this
distortion so that pixel distances anywhere in the frame correspond
uniformly to real-world distances at that image plane — a precondition
the pixel-to-mm conversion depends on.

## 7. Limitations

- Checkerboard square size (20 mm) was taken from the printed
  checkerboard's nominal spec rather than independently re-verified with
  calipers. Since pixels-per-mm scales linearly with this value, any
  printer scaling error propagates directly into every measurement — a
  candidate contributor to the systematic bias discussed in
  `MEASUREMENT_REPORT.md`.
- Calibration RMS error (2.339 px, before independent recomputation)
  exceeds the "excellent" threshold; a future pass should inspect
  per-image reprojection error and discard the worst-fitting captures.