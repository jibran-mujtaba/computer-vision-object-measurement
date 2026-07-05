## Setup & Installation Guide

### Environment
Operating System: Windows
Development Environment: Visual Studio Code
Python Environment: Local Python virtual environment (venv)
Hardware Used: CPU
GPU: Not used. The project was developed and executed on a CPU-only system.
Docker: Not used. The project runs entirely in a local Python environment.
1. Clone the Repository and Create a Virtual Environment
git clone <https://github.com/jibran-mujtaba/computer-vision-object-measurement>
cd computer-vision-object-measurement

# Windows

2. Install Dependencies

Install all required Python packages:

pip install -r requirements.txt
requirements.txt
opencv-python
numpy
matplotlib
pandas
tqdm
Pillow
torch
torchvision
scikit-learn

3. Camera Calibration

Camera calibration only needs to be performed once for each camera and lens combination.

Place at least 20 checkerboard images (9 × 7 inner corners) inside the calibration image directory and run:

python calibration/images/calibrate.py

This generates:

calibration/images/calibration_data.npz

The calibration file stores:

Camera intrinsic matrix
Distortion coefficients
Calibration image resolution

These parameters are required for accurate image undistortion and real-world measurement.

To visually verify the calibration on an image, run:

python calibration/images/undistort.py
4. Dataset Preparation

The project expects the dataset to be organized as follows:

dataset/
│
├── train/
│   ├── images/
│   ├── masks/
│   └── annotations.json
│
├── val/
│   ├── images/
│   ├── masks/
│   └── annotations.json
│
└── test/
    ├── images/
    ├── masks/
    └── annotations.json

Each image must have a corresponding segmentation mask with the same filename (excluding the extension).

5. Train the Segmentation Model

Run the training script:

python models/train.py

Training outputs are automatically saved to:

outputs/
├── checkpoints/
│   └── best_model.pth
│
├── training_log.csv
│
└── loss_curve.png

Where:

best_model.pth – Model with the highest validation IoU.
training_log.csv – Training and validation metrics recorded for every epoch.
loss_curve.png – Visualization of training loss and validation IoU over time.

Training hyperparameters (such as epochs, learning rate, and batch size) are defined as constants within models/train.py.

6. Run Object Measurement
Measure a Single Image
python measurement/measure.py "path/to/image.jpg"
Measure an Entire Folder
python measurement/measure.py "dataset/test/images"

The measurement pipeline automatically:

Loads the trained segmentation model.
Performs image segmentation.
Detects the checkerboard.
Corrects camera distortion.
Computes real-world object dimensions.
Saves measurement results.

Generated outputs are stored in:

outputs/
└── measurement_results/
    ├── annotated images
    ├── segmentation masks
    └── trials.csv

Intermediate processing files are also generated for debugging purposes when needed.

The trained model (best_model.pth) is automatically loaded by measurement/predictor.py; no manual model-loading step is required.

7. Validate Measurement Accuracy

After updating the measured object's actual dimensions inside trials.csv, run:

python measurement/validate_accuracy.py "outputs/measurement_results/trials.csv"

The script computes measurement performance metrics, including:

Error for each measurement trial
Mean Absolute Error (MAE)
Mean Percentage Error (MPE)

These metrics are printed in the console after execution.

Project Workflow

The overall execution pipeline is:

Camera Calibration
        │
        ▼
Dataset Preparation
        │
        ▼
Model Training
        │
        ▼
Model Checkpoint (best_model.pth)
        │
        ▼
Segmentation Inference
        │
        ▼
Checkerboard Detection
        │
        ▼
Pixel-to-Millimeter Conversion
        │
        ▼
Object Measurement
        │
        ▼
Measurement Validation
Project Outputs
Output	Description
calibration/images/calibration_data.npz	Camera calibration parameters
outputs/checkpoints/best_model.pth	Best trained segmentation model
outputs/training_log.csv	Training and validation metrics
outputs/loss_curve.png	Training loss and validation IoU curves
outputs/measurement_results/	Measured images, masks, and results
outputs/measurement_results/trials.csv	Measurement results for validation