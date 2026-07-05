# Model Training Report

## 1. Architecture

- **Model:** custom **U-Net** implemented from scratch (`models/unet.py`),
  `n_channels=3` (RGB input), `n_classes=1` (binary mask output, single
  `laptop` class).
- **Framework:** PyTorch.
- **Training entry point:** `models/train.py`.
- **Why U-Net over a pretrained detector / YOLO / Roboflow model:** the
  measurement pipeline needs a pixel-accurate object *boundary*, not a
  bounding box — U-Net's encoder-decoder structure with skip connections
  is a standard, well-understood choice for exactly this kind of dense,
  single-object binary segmentation, and training it from scratch on a
  small, focused single-class dataset avoids the overhead/mismatch of
  adapting a large pretrained detection backbone to a task it wasn't
  designed for.

## 2. Training setup

| Hyperparameter    | Value                                                     |
|-------------------|-----------------------------------------------------------|
| Epochs            | 50                                                        |
| Batch size        | 4                                                         |
| Optimizer         | Adam                                                      |
| Learning rate     | 1e-4                                                      |
| LR scheduler      | `ReduceLROnPlateau` (mode=`min`, factor=0.5, patience=5)  |
| Loss function     | `BCEWithLogitsLoss` + Dice Loss (summed)                  |
| Mixed precision   | `torch.cuda.amp` (enabled when CUDA available)            |
| Input image size  | 640×640                                                   |
| Train / Val images| 57 / 15                                                   |
| Seed              | 42                                                        |

**Loss function rationale:** BCE alone tends to be dominated by the
(larger) background class in object-vs-background segmentation; Dice loss
directly optimizes region overlap and is far less sensitive to class
imbalance. Combining both gives stable pixel-level gradients (from BCE)
alongside a mask-overlap-aware signal (from Dice).

**Metric choice — IoU/Dice/Precision/Recall, not mAP:** mAP
(`mAP@0.5` / `mAP@0.5:0.95`) is an object-detection metric, built around
ranking multiple candidate boxes/masks per image by confidence and
integrating precision-recall across IoU thresholds and, typically,
multiple classes. This task is single-class, single-mask-per-image binary
segmentation — there is no multi-object ranking problem to evaluate. IoU
and Dice are the direct, standard measures of segmentation mask quality
for this setup, supplemented with pixel-wise Precision/Recall; these are
reported below in place of mAP.

## 3. Training results

Full per-epoch log: `outputs/training_log.csv`. Loss/IoU curves:
`outputs/loss_curve.png`.

**Best checkpoint** (highest validation IoU, saved to
`outputs/checkpoints/best_model.pth`):

| Metric          | Value     | Epoch |
|-----------------|-----------|-------|
| Validation IoU  | **0.8902**| 44    |
| Validation Dice | 0.9356    | 44    |
| Validation Loss | 0.6500    | 44    |
| Train IoU       | 0.9018    | 44    |
| Train Loss      | 0.6321    | 44    |
| Precision       | 0.9476    | 44    |
| Recall          | 0.9628    | 44    |

**Final epoch (50)** — training continued past the best checkpoint, as
configured (50 total epochs, no early stopping):

| Metric          | Value  |
|-----------------|--------|
| Train Loss      | 0.5818 |
| Validation Loss | 0.6393 |
| Train IoU       | 0.9544 |
| Validation IoU  | 0.8678 |
| Train Dice      | 0.9761 |
| Validation Dice | 0.9140 |
| Precision       | 0.9352 |
| Recall          | 0.9541 |

**Training curve summary (epoch 1 → 50):**
- Train loss decreased steadily from 1.375 → 0.582 across training.
- Validation IoU was noisy epoch-to-epoch (expected with only 15
  validation images and batch size 4) but trended upward overall, from
  0.168 at epoch 1 to a peak of 0.890 at epoch 44.
- Validation loss/IoU show visible oscillation after ~epoch 30 (e.g.
  epoch 29 val IoU dips to 0.670 immediately after epoch 28's 0.784),
  consistent with a small validation set combined with a plateau-based LR
  scheduler that hadn't yet triggered a reduction at that point.
- The gap between train IoU (0.9544 by epoch 50) and validation IoU
  (0.8678 at epoch 50, versus 0.8902 at the best checkpoint) indicates
  mild overfitting in later epochs — the best checkpoint at epoch 44 was
  correctly selected over the final epoch's weights for this reason.


## 4. Inference pipeline

`measurement/predictor.py::Predictor` loads the trained checkpoint
(`outputs/checkpoints/best_model.pth`) and exposes a `predict()` method
used by `measurement/measure.py::get_segmentation()`. Inference always
runs on the **undistorted** image; the raw model output is cleaned with
morphological closing then opening (5×5 kernel) before being passed to
`measure_object()` for the pixel-to-mm conversion — see
`MEASUREMENT_REPORT.md`.

## 5. Limitations

- **640×640 fixed-resolution training input** does not preserve the
  original image aspect ratio (see `DATASET_CARD.md`); this is assessed
  as the primary driver of the directional measurement bias in
  `MEASUREMENT_REPORT.md`, and should be corrected (aspect-preserving
  resize/letterboxing) before any further training run.
- Validation set is small (15 images), which explains the epoch-to-epoch
  noise in validation metrics; a larger validation split would give a
  more stable signal for model/checkpoint selection.
- No formal hyperparameter search was performed (learning rate, batch
  size, and loss weighting were fixed choices, not tuned via sweep).