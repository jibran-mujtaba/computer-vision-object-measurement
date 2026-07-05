# Computer Vision Object Measurement System

An end-to-end computer vision pipeline that measures the real-world width
and height (in millimetres) of a physical object — a **laptop** — from a
single photograph, using camera calibration and a trained segmentation
model.

## What it does

1. **Calibrates** a camera's intrinsic parameters using checkerboard
   images (`calibration/`), removing lens distortion (radial + tangential).
2. **Segments** the target object out of a photo using a custom U-Net
   trained on a self-collected, self-labelled dataset (`models/`).
3. **Measures** the object in millimetres by converting the segmentation
   mask's pixel dimensions into real-world units, using a checkerboard of
   known square size as an in-frame metric reference (`measurement/`).
4. **Validates** accuracy against physical ruler/caliper ground truth
   (`measurement/validate_accuracy.py`).

## Repository structure

```
computer-vision-object-measurement/
├── calibration/
│   ├── images/
│   │   ├── calibrate.py              # intrinsic calibration script
│   │   ├── calibration_data.npz      # camera matrix, distortion, calib resolution
│   │   └── undistort.py              # standalone visual undistortion check (see CALIBRATION_REPORT.md)
│   └── rough_test.ipynb              # exploratory scratch notebook
├── dataset/
│   ├── annotations.json              # full COCO-segmentation export (Roboflow)
│   ├── train/annotations.json
│   ├── val/annotations.json
│   └── test/annotations.json
├── models/
│   ├── unet.py                       # U-Net architecture definition
│   └── train.py                      # training entry point
├── measurement/
│   ├── measure.py                    # undistort → detect checkerboard → segment → measure
│   ├── predictor.py                  # loads trained U-Net, runs inference
│   └── validate_accuracy.py          # MAE/MPE from a filled-in trials.csv
├── outputs/
│   ├── checkpoints/best_model.pth
│   ├── training_log.csv
│   ├── loss_curve.png
│   └── measurement_results/
│       ├── trials.csv
│       └── debug/                    # per-run intermediate images
├── docs/
│   ├── CALIBRATION_REPORT.md
│   ├── DATASET_CARD.md
│   ├── TRAINING_REPORT.md
│   ├── MEASUREMENT_REPORT.md
│   └── SETUP.md
├── requirements.txt
└── README.md
```

## Pipeline architecture

```
   Raw photo (object + checkerboard in same frame)
              │
              ▼
   cv2.undistort()  ◄── camera_matrix, distortion  (calibration/images/calibration_data.npz)
      (rescaled to match input image resolution — see CALIBRATION_REPORT.md)
              │
              ▼
   ┌──────────────────────┬───────────────────────────┐
   │ Checkerboard detect   │  U-Net segmentation        │
   │ → pixels_per_mm ratio │  → binary object mask      │
   └──────────────────────┴───────────────────────────┘
              │                        │
              ▼                        ▼
        pixels_per_mm     ×     mask contour (minAreaRect)
              │                        │
              └────────────┬───────────┘
                           ▼
              Width (mm), Height (mm), confidence
                           │
                           ▼
              Annotated output image + trials.csv row
```

Undistortion is applied **before** both checkerboard detection and
segmentation — every pixel measurement in this system is made on a
lens-corrected image. See `docs/CALIBRATION_REPORT.md` for why this is
mandatory.

## Quick start

Full details in `docs/SETUP.md`. Short version:

pip install -r requirements.txt

# 1. Calibrate the camera (only needed once per camera/lens)
python calibration/images/calibrate.py

# 2. Train the segmentation model
python models/train.py

# 3. Measure a single image
python measurement/measure.py "path/to/photo.jpg"

# 3b. Or batch-measure a whole folder
python measurement/measure.py "dataset/test/images"

# 4. Validate accuracy against physical ground truth
python measurement/validate_accuracy.py "outputs/measurement_results/trials.csv"


## Documentation index


| `docs/CALIBRATION_REPORT.md` | Calibration method, intrinsic matrix, distortion coefficients, reprojection error |

| `docs/DATASET_CARD.md` | Object justification, collection method, labelling tool, dataset statistics |

| `docs/TRAINING_REPORT.md` | Model architecture, hyperparameters, training/validation metrics, loss curves |

| `docs/MEASUREMENT_REPORT.md` | Pixel-to-mm derivation, accuracy validation, error analysis, limitations |

| `docs/SETUP.md` | Environment, installation, run instructions |

## Design decisions (summary — full detail in linked reports)

- **Custom U-Net over a pretrained/off-the-shelf detector**: this is a
  single-class, pixel-accurate boundary problem (the object's outline
  drives the mm measurement directly), so a segmentation network trained
  specifically on the target object was chosen over a general-purpose
  detector.
- **Checkerboard-in-frame reference** rather than a separately calibrated
  reference per photo: the same image used for measurement contains the
  metric reference, so the pixels-per-mm ratio is computed at the same
  distance and plane as the object being measured.
- **IoU/Dice/Precision/Recall over mAP** for evaluating the segmentation
  model: mAP is an object-detection metric built around ranking multiple
  bounding-box/mask proposals across confidence and IoU thresholds; this
  is a single-class, single-mask-per-image binary segmentation task, so
  IoU and Dice are the direct, standard measures of mask quality — see
  `docs/TRAINING_REPORT.md`.

## Known limitations

Summarized here; full discussion in `docs/MEASUREMENT_REPORT.md` and
`docs/DATASET_CARD.md`:

- The Roboflow export step resized all training images to a fixed
  **640×640** square, which does not preserve the original aspect ratio
  of the (non-square) laptop images. This is identified as the leading
  cause of the height-measurement bias documented in the accuracy report.
- Camera calibration images were resized (`RESIZE_SCALE = 0.5`) before
  running `cv2.calibrateCamera`, so the resulting camera matrix is only
  valid at that resolution — the measurement pipeline rescales the matrix
  before applying it to full-resolution photos
  (`measure.py::scale_camera_matrix()`); see `docs/CALIBRATION_REPORT.md`.

## Repository housekeeping notes

A few non-essential files currently sit in the tree and are called out here
for transparency rather than silently cleaned up:

- `calibration/calibration_data.npz` (root-level) and
  `calibration/images/calibration_data.npz` — two calibration files exist;
  only the one in `calibration/images/` is used by `measure.py`
  (`CALIBRATION_FILE` points there explicitly). The root-level copy is a
  leftover from an earlier debugging session (see `docs/CALIBRATION_REPORT.md`)
  and should be removed before final submission to avoid ambiguity about
  which calibration is authoritative.
- `calibration/images/tempCodeRunnerFile.py`,
  `measurement/tempCodeRunnerFile.py` — auto-generated by the VS Code Code
  Runner extension, not part of the pipeline.
- `calibration/rough_test.ipynb` — exploratory scratch notebook, not part
  of the production pipeline.
- `measurement/__pycache__/`, `models/__pycache__/` — compiled bytecode.