
import cv2
import numpy as np
import glob
import os
import time

# ==========================
# Configuration
# ==========================
CHECKERBOARD = (9, 7)      # INNER corners (columns, rows)
SQUARE_SIZE = 20           # mm
SHOW_IMAGES = False        # True if you want to visualize corners
RESIZE_SCALE = 0.5         # Reduce image size for faster detection

# ==========================
# Prepare object points
# ==========================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# ==========================
# Load images
# ==========================
script_dir = os.path.dirname(os.path.abspath(__file__))

image_patterns = [
    "*.jpg",
    "*.jpeg",
    "*.png"
]

images = []
for pattern in image_patterns:
    images.extend(glob.glob(os.path.join(script_dir, pattern)))

print("=" * 60)
print("Camera Calibration Started")
print("=" * 60)
print("Script directory:", script_dir)
print(f"Found {len(images)} images")

if len(images) == 0:
    print("ERROR: No calibration images found.")
    exit()

gray_shape = None
successful_images = 0

# ==========================
# Process images
# ==========================
for idx, fname in enumerate(images):
    print("\n" + "-" * 50)
    print(f"[{idx+1}/{len(images)}] Processing: {os.path.basename(fname)}")

    start = time.time()

    img = cv2.imread(fname)

    if img is None:
        print("ERROR: Failed to read image.")
        continue

    print("Original shape:", img.shape)

    if RESIZE_SCALE != 1.0:
        img = cv2.resize(
            img,
            None,
            fx=RESIZE_SCALE,
            fy=RESIZE_SCALE
        )
        print("Resized shape:", img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray_shape is None:
        gray_shape = gray.shape[::-1]

    print("Running checkerboard detection...")

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK
    )

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags
    )

    print("Detection result:", ret)

    if ret:
        successful_images += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        imgpoints.append(corners2)

        print("Corners refined successfully.")

        if SHOW_IMAGES:
            cv2.drawChessboardCorners(
                img,
                CHECKERBOARD,
                corners2,
                ret
            )
            cv2.imshow("Corners", img)
            cv2.waitKey(300)
    else:
        print("Checkerboard NOT detected.")

    end = time.time()
    print(f"Time taken: {end-start:.2f} sec")

cv2.destroyAllWindows()

print("\n" + "=" * 60)
print(f"Valid calibration images: {successful_images}/{len(images)}")
print("=" * 60)

if len(objpoints) == 0:
    print("ERROR: No checkerboards detected in any image.")
    print("Check:")
    print("1. Correct checkerboard INNER corners")
    print("2. Board visible in images")
    print("3. Good lighting / no blur")
    exit()

# ==========================
# Camera Calibration
# ==========================
print("\nRunning camera calibration...")

ret, cameraMatrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray_shape,
    None,
    None
)

print("\nCalibration RMS Error:", ret)

print("\nCamera Matrix:")
print(cameraMatrix)

print("\nDistortion Coefficients:")
print(distortion)

# ==========================
# Reprojection Error
# ==========================
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        cameraMatrix,
        distortion
    )

    error = cv2.norm(
        imgpoints[i],
        imgpoints2,
        cv2.NORM_L2
    ) / len(imgpoints2)

    mean_error += error

mean_error /= len(objpoints)

print("\nMean Reprojection Error:", mean_error)

# ==========================
# Save calibration
# ==========================
save_path = os.path.join(script_dir, "calibration_data.npz")

np.savez(
    save_path,
    cameraMatrix=cameraMatrix,
    distortion=distortion
)

print("\nCalibration saved to:")
print(save_path)

print("\nDone.")