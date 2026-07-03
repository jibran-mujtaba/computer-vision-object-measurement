
import cv2
import numpy as np
import os

# =====================================
# Configuration
# =====================================
DISPLAY_SCALE = 0.12      # For screen display only
CROP_OUTPUT = False       # True to crop black borders after undistortion
CALIBRATION_SCALE = 2.0   # Because calibration used resized images (0.5)

# =====================================
# Paths
# =====================================
script_dir = os.path.dirname(os.path.abspath(__file__))

calibration_path = os.path.join(
    script_dir,
    "calibration_data.npz"
)

image_path = os.path.abspath(
    os.path.join(
        script_dir,
        "../../dataset/laptop.jpeg"
    )
)

print("=" * 60)
print("Undistortion Script Started")
print("=" * 60)

print("Calibration file:", calibration_path)
print("Calibration exists:", os.path.exists(calibration_path))

print("Image file:", image_path)
print("Image exists:", os.path.exists(image_path))

# =====================================
# Load calibration
# =====================================
data = np.load(calibration_path)

cameraMatrix = data["cameraMatrix"]
distortion = data["distortion"]

print("\nLoaded Camera Matrix:")
print(cameraMatrix)

print("\nLoaded Distortion:")
print(distortion)

# =====================================
# Load image
# =====================================
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

h, w = img.shape[:2]

print("\nOriginal image shape:", img.shape)

# =====================================
# Scale camera matrix to original resolution
# =====================================
cameraMatrix_scaled = cameraMatrix.copy()

cameraMatrix_scaled[0, 0] *= CALIBRATION_SCALE
cameraMatrix_scaled[1, 1] *= CALIBRATION_SCALE
cameraMatrix_scaled[0, 2] *= CALIBRATION_SCALE
cameraMatrix_scaled[1, 2] *= CALIBRATION_SCALE

print("\nScaled Camera Matrix:")
print(cameraMatrix_scaled)

# =====================================
# Compute optimal new camera matrix
# =====================================
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix_scaled,
    distortion,
    (w, h),
    1,
    (w, h)
)

print("\nNew Camera Matrix:")
print(newCameraMatrix)

print("\nROI:", roi)

# =====================================
# Undistort
# =====================================
undistorted = cv2.undistort(
    img,
    cameraMatrix_scaled,
    distortion,
    None,
    newCameraMatrix
)

print("Undistorted shape before crop:", undistorted.shape)

# =====================================
# Crop (optional)
# =====================================
if CROP_OUTPUT:
    x, y, rw, rh = roi

    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]
        print("Undistorted shape after crop:", undistorted.shape)
    else:
        print("Invalid ROI. Skipping crop.")

# =====================================
# Save outputs
# =====================================
output_original = os.path.join(script_dir, "original_output.jpg")
output_undistorted = os.path.join(script_dir, "undistorted_output.jpg")

cv2.imwrite(output_original, img)
cv2.imwrite(output_undistorted, undistorted)

print("\nSaved files:")
print(output_original)
print(output_undistorted)

# =====================================
# Resize for display (to avoid huge windows)
# =====================================
display_original = cv2.resize(
    img,
    None,
    fx=DISPLAY_SCALE,
    fy=DISPLAY_SCALE
)

display_undistorted = cv2.resize(
    undistorted,
    None,
    fx=DISPLAY_SCALE,
    fy=DISPLAY_SCALE
)

print("\nDisplay original shape:", display_original.shape)
print("Display undistorted shape:", display_undistorted.shape)

# =====================================
# Show
# =====================================
cv2.imshow("Original (Scaled for Display)", display_original)
cv2.imshow("Undistorted (Scaled for Display)", display_undistorted)

print("\nPress any key inside image window to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nDone.")

