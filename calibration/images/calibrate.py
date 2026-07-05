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
SHOW_IMAGES = False
RESIZE_SCALE = 0.5

# Force every image into this orientation before processing.
# "landscape" = width > height. Change to "portrait" if you'd
# rather standardize the other way — just be consistent with
# measure.py.
CANONICAL_ORIENTATION = "landscape"


def normalize_orientation(img):
    """
    Rotate image 90 degrees if needed so every image shares the
    same width>height (or height>width) convention. Calibration
    requires all images to share a single, consistent imageSize —
    mixing portrait/landscape captures silently corrupts the
    resulting camera matrix.
    """
    h, w = img.shape[:2]
    is_landscape = w > h

    if CANONICAL_ORIENTATION == "landscape" and not is_landscape:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if CANONICAL_ORIENTATION == "portrait" and is_landscape:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img


# ==========================
# Prepare object points
# ==========================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# ==========================
# Load images
# ==========================
script_dir = os.path.dirname(os.path.abspath(__file__))

image_patterns = ["*.jpg", "*.jpeg", "*.png"]

images = []
for pattern in image_patterns:
    images.extend(glob.glob(os.path.join(script_dir, pattern)))

# Exclude leftover test/output images that aren't real calibration
# captures (e.g. from previous undistort.py runs).
EXCLUDE_NAMES = {"original_output.jpg", "undistorted_output.jpg"}
images = [f for f in images if os.path.basename(f) not in EXCLUDE_NAMES]

print("=" * 60)
print("Camera Calibration Started")
print("=" * 60)
print("Script directory:", script_dir)
print(f"Found {len(images)} calibration images (after exclusions)")
print(f"Canonical orientation: {CANONICAL_ORIENTATION}")

if len(images) == 0:
    print("ERROR: No calibration images found.")
    exit()

gray_shape = None
successful_images = 0
orientation_corrections = 0

for idx, fname in enumerate(images):
    print("\n" + "-" * 50)
    print(f"[{idx+1}/{len(images)}] Processing: {os.path.basename(fname)}")

    start = time.time()

    img = cv2.imread(fname)

    if img is None:
        print("ERROR: Failed to read image.")
        continue

    print("Original shape:", img.shape)

    orig_h, orig_w = img.shape[:2]
    was_landscape = orig_w > orig_h

    img = normalize_orientation(img)

    new_h, new_w = img.shape[:2]
    if (new_w > new_h) != was_landscape:
        orientation_corrections += 1
        print(f"  Rotated to canonical orientation -> shape now {img.shape}")

    if RESIZE_SCALE != 1.0:
        img = cv2.resize(img, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        print("Resized shape:", img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Running checkerboard detection...")

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK
    )

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    print("Detection result:", ret)

    if ret:
        # Only lock gray_shape in from an image that actually
        # succeeded AND has already been orientation-normalized —
        # guarantees every accepted image shares this exact frame.
        if gray_shape is None:
            gray_shape = gray.shape[::-1]
            print(f"  Locked calibration imageSize (w,h) = {gray_shape}")
        elif gray.shape[::-1] != gray_shape:
            print(
                f"  ERROR: this image's shape {gray.shape[::-1]} doesn't match "
                f"locked calibration size {gray_shape} even after orientation "
                f"normalization. Skipping this image — check for a differently "
                f"sized/cropped source photo."
            )
            continue

        successful_images += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        imgpoints.append(corners2)
        print("Corners refined successfully.")

        if SHOW_IMAGES:
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow("Corners", img)
            cv2.waitKey(300)
    else:
        print("Checkerboard NOT detected.")

    end = time.time()
    print(f"Time taken: {end-start:.2f} sec")

cv2.destroyAllWindows()

print("\n" + "=" * 60)
print(f"Valid calibration images: {successful_images}/{len(images)}")
print(f"Images requiring orientation correction: {orientation_corrections}")
print("=" * 60)

if len(objpoints) == 0:
    print("ERROR: No checkerboards detected in any image.")
    exit()

print("\nRunning camera calibration...")

ret, cameraMatrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray_shape, None, None
)

print("\nCalibration RMS Error:", ret)
print("\nCamera Matrix:")
print(cameraMatrix)
print("\nDistortion Coefficients:")
print(distortion)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distortion
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
mean_error /= len(objpoints)

print("\nMean Reprojection Error:", mean_error)

if ret > 1.0:
    print(
        "\n  WARNING: RMS error is high (>1.0 px). Even after fixing orientation "
        "mixing, this suggests some images may have poor corner detection quality "
        "(motion blur, glare, oblique angle). Consider inspecting per-image "
        "reprojection error and discarding the worst offenders."
    )

save_path = os.path.join(script_dir, "calibration_data.npz")

np.savez(
    save_path,
    cameraMatrix=cameraMatrix,
    distortion=distortion,
    calibrationImageSize=np.array(gray_shape),  # (w, h) of images calibration was fit on
    calibrationOrientation=CANONICAL_ORIENTATION
)

print("\nCalibration saved to:")
print(save_path)
print("Calibration resolution (w, h):", gray_shape)
print("Canonical orientation:", CANONICAL_ORIENTATION)
print(
    "\nIMPORTANT: measure.py must load THIS EXACT FILE "
    f"({save_path}) and must rescale the camera matrix to match "
    "whatever resolution the input photo actually is — the matrix "
    "above is only valid at the resolution printed above."
)
print("\nDone.")