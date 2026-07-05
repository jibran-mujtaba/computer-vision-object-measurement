import os
import csv
import cv2
import numpy as np

from predictor import Predictor

# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# NOTE: calibrate.py saves to <script_dir>/calibration_data.npz, and
# calibrate.py itself lives in calibration/images/. This path MUST
# point at the exact same file calibrate.py wrote, or you'll silently
# load a stale/unrelated calibration.
CALIBRATION_FILE = os.path.join(
    PROJECT_ROOT,
    "calibration",
    "images",
    "calibration_data.npz"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "measurement_results"
)

DEBUG_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "debug"
)

TRIALS_CSV_PATH = os.path.join(OUTPUT_FOLDER, "trials.csv")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

CHECKERBOARD = (9, 7)   # inner corners (cols, rows)
SQUARE_SIZE = 20.0      # millimetres

MAX_ROI_AREA_FRACTION = 0.5
MIN_ROI_AREA = 4000

# Detection resize ceiling. Kept generous — downscaling too
# aggressively softens fine checkerboard corners and can be
# the actual cause of "detection fails in pipeline but works
# on raw image" symptoms.
MAX_DETECT_SIZE = 2600

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _debug_save(name, image):
    path = os.path.join(DEBUG_FOLDER, name)
    cv2.imwrite(path, image)
    print(f"  [debug saved] {path}")


print("Loading trained U-Net model...")
predictor = Predictor()
print("Model loaded successfully.\n")


# ==========================================================
# LOAD CAMERA CALIBRATION
# ==========================================================

def load_calibration():

    if not os.path.exists(CALIBRATION_FILE):
        raise FileNotFoundError(
            "\nCalibration file not found.\n\n"
            f"Expected:\n{CALIBRATION_FILE}\n\n"
            "Run calibration/images/calibrate.py first."
        )

    print("Loading camera calibration...")
    data = np.load(CALIBRATION_FILE)
    camera_matrix = data["cameraMatrix"]
    distortion = data["distortion"]

    if "calibrationImageSize" in data:
        calib_w, calib_h = data["calibrationImageSize"]
    else:
        raise RuntimeError(
            "\nThis calibration file has no 'calibrationImageSize' entry, "
            "so the camera matrix cannot be safely rescaled to match the "
            "input image's resolution. Re-run calibrate.py to regenerate it."
        )

    print("Camera matrix (as calibrated):\n", camera_matrix)
    print("Distortion coefficients:\n", distortion)
    print(f"Calibration was computed at resolution (w,h) = ({calib_w}, {calib_h})")
    print("Calibration loaded successfully.")

    return camera_matrix, distortion, (int(calib_w), int(calib_h))


def scale_camera_matrix(camera_matrix, calib_size, actual_size):
    """
    Rescales fx, fy, cx, cy proportionally to the actual image size, since
    a camera matrix calibrated at one resolution is not directly valid at
    another. Distortion coefficients are dimensionless and untouched.
    """
    calib_w, calib_h = calib_size
    actual_w, actual_h = actual_size

    sx = actual_w / float(calib_w)
    sy = actual_h / float(calib_h)

    scaled = camera_matrix.copy()
    scaled[0, 0] *= sx   # fx
    scaled[0, 2] *= sx   # cx
    scaled[1, 1] *= sy   # fy
    scaled[1, 2] *= sy   # cy

    print(
        f"  Rescaling camera matrix: calib=({calib_w}x{calib_h}) -> "
        f"actual=({actual_w}x{actual_h})  (sx={sx:.4f}, sy={sy:.4f})"
    )
    print("  Scaled camera matrix:\n", scaled)

    return scaled


# ==========================================================
# UNDISTORT IMAGE
# ==========================================================
def _sanity_check_calibration(camera_matrix, image_shape):
    h, w = image_shape[:2]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    cx_ratio = cx / w
    cy_ratio = cy / h

    print(f"  Calibration sanity check: cx/width={cx_ratio:.3f}, "
          f"cy/height={cy_ratio:.3f} (expect ~0.4-0.6 for both)")

    if not (0.35 <= cx_ratio <= 0.65) or not (0.35 <= cy_ratio <= 0.65):
        print(
            "  *** WARNING: principal point is far from image center "
            "relative to this image's dimensions. This calibration "
            "likely does not match this image's resolution/aspect "
            "ratio, or the calibration itself converged poorly. "
            "Undistortion results are NOT trustworthy. ***"
        )
        return False

    return True


def undistort_image(image, camera_matrix, distortion):

    h, w = image.shape[:2]

    _sanity_check_calibration(camera_matrix, image.shape)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, (w, h), 1, (w, h)
    )

    undistorted = cv2.undistort(
        image, camera_matrix, distortion, None, new_camera_matrix
    )

    x, y, rw, rh = roi

    print(f"  Undistortion ROI: x={x}, y={y}, w={rw}, h={rh} (full frame: {w}x{h})")

    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y + rh, x:x + rw]
    else:
        print("  WARNING: undistortion ROI is degenerate (0 width/height). "
              "This usually means the calibration is a poor fit for this "
              "image resolution. Skipping crop, using full undistorted frame.")

    return undistorted


# ==========================================================
# PREPROCESSING VARIANTS
# ==========================================================

def _resize_for_detection(image):
    height, width = image.shape[:2]
    longest = max(height, width)
    scale = 1.0

    if longest > MAX_DETECT_SIZE:
        scale = MAX_DETECT_SIZE / float(longest)
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    return image, scale


def _make_gray_variants(image, tag=""):
    resized, scale = _resize_for_detection(image)

    gray_raw = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray_raw)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_raw)

    gray_eq_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    variants = [
        ("plain_gray", gray_raw),
        ("equalized", gray_eq),
        ("clahe", gray_clahe),
        ("equalized_blur", gray_eq_blur),
    ]

    for label, g in variants:
        _debug_save(f"{tag}_{label}.jpg", g)

    return variants, scale


# ==========================================================
# CANDIDATE PATTERN SEARCH
# ==========================================================

def _try_patterns(gray, label=""):

    candidate_patterns = [CHECKERBOARD, (CHECKERBOARD[1], CHECKERBOARD[0])]

    for pattern in candidate_patterns:

        print(f"    [{label}] trying pattern {pattern} ...", end=" ")

        found = False
        corners = None

        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                found, corners = cv2.findChessboardCornersSB(
                    gray, pattern,
                    flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                )
            except cv2.error as e:
                print(f"SB error ({e})", end=" ")
                found = False

        if not found:
            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE |
                cv2.CALIB_CB_FAST_CHECK
            )
            found, corners = cv2.findChessboardCorners(gray, pattern, flags)

        print("FOUND" if found else "not found")

        if found:
            return True, corners, pattern

    return False, None, None


def _try_all_variants(image, tag):
    variants, scale = _make_gray_variants(image, tag)

    for label, gray in variants:
        found, corners, pattern = _try_patterns(gray, label=f"{tag}:{label}")
        if found:
            return True, corners, pattern, scale, label

    return False, None, None, scale, None


# ==========================================================
# ROI LOCATION (corner-density based fallback)
# ==========================================================

def locate_checkerboard_region(image):

    print("\n  Locating checkerboard ROI (corner density method)...")

    resized, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    image_area = resized.shape[0] * resized.shape[1]

    edges = cv2.Canny(gray_eq, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    _debug_save("roi_edges.jpg", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_roi = None
    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_ROI_AREA or area > MAX_ROI_AREA_FRACTION * image_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / float(h)

        if ratio < 0.5 or ratio > 2.0:
            continue

        patch = gray_eq[y:y + h, x:x + w]
        features = cv2.goodFeaturesToTrack(
            patch, maxCorners=300, qualityLevel=0.05, minDistance=5
        )
        corner_count = 0 if features is None else len(features)
        score = corner_count / float(area)

        if score > best_score:
            best_score = score
            best_roi = (x, y, w, h)

    if best_roi is None:
        print("  No ROI candidate found. Using full image.")
        return image, (0, 0)

    x, y, w, h = best_roi
    padding = 40

    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(resized.shape[1] - x, w + padding * 2)
    h = min(resized.shape[0] - y, h + padding * 2)

    inv = 1.0 / scale if scale != 1 else 1.0

    ox = int(x * inv)
    oy = int(y * inv)
    ow = min(int(w * inv), image.shape[1] - ox)
    oh = min(int(h * inv), image.shape[0] - oy)

    roi = image[oy:oy + oh, ox:ox + ow]

    print(f"  ROI found at original-image coords: x={ox}, y={oy}, w={ow}, h={oh}")
    _debug_save("roi_crop.jpg", roi)

    return roi, (ox, oy)


# ==========================================================
# DETECT CHECKERBOARD (main entry point, fully instrumented)
# ==========================================================

def detect_checkerboard(image, image_tag="undistorted"):

    print(f"\n{'='*60}")
    print(f"Checkerboard detection on: {image_tag} image")
    print(f"Image shape: {image.shape}")
    print(f"{'='*60}")

    _debug_save(f"{image_tag}_input.jpg", image)

    print(f"\n[Step 1] Full-image detection ({image_tag})...")
    found, corners, pattern, scale, variant = _try_all_variants(image, f"{image_tag}_full")

    offset_x, offset_y = 0, 0

    if found:
        print(f"  -> SUCCESS on full image using '{variant}' preprocessing, pattern {pattern}")

    if not found:
        print(f"\n[Step 2] Full-image detection failed. Trying ROI isolation ({image_tag})...")
        roi, (offset_x, offset_y) = locate_checkerboard_region(image)
        found, corners, pattern, scale, variant = _try_all_variants(roi, f"{image_tag}_roi")

        if found:
            print(f"  -> SUCCESS on ROI crop using '{variant}' preprocessing, pattern {pattern}")

    if not found:
        print(f"\n  [RESULT] Checkerboard NOT found in {image_tag} image, any variant, any pattern.")
        return False, None, None, None

    if offset_x == 0 and offset_y == 0:
        variants, _ = _make_gray_variants(image, f"{image_tag}_refine_full")
    else:
        roi, _ = locate_checkerboard_region(image)
        variants, _ = _make_gray_variants(roi, f"{image_tag}_refine_roi")

    gray_for_refine = dict(variants)[variant]

    corners = cv2.cornerSubPix(
        gray_for_refine, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    inv_scale = 1.0 / scale if scale != 1 else 1.0
    corners = corners * inv_scale

    corners[:, :, 0] += offset_x
    corners[:, :, 1] += offset_y

    debug = image.copy()
    cv2.drawChessboardCorners(debug, pattern, corners, True)
    _debug_save(f"{image_tag}_corners_detected.jpg", debug)

    print(f"\n  [RESULT] Checkerboard FOUND: pattern={pattern}, variant={variant}")

    return True, corners, pattern, variant


# ==========================================================
# PIXELS PER MILLIMETRE
# ==========================================================

def calculate_pixels_per_mm(corners, detected_pattern):

    cols, rows = detected_pattern

    horizontal = []
    vertical = []

    for r in range(rows):
        for c in range(cols - 1):
            p1 = corners[r * cols + c][0]
            p2 = corners[r * cols + c + 1][0]
            horizontal.append(np.linalg.norm(p1 - p2))

    for c in range(cols):
        for r in range(rows - 1):
            p1 = corners[r * cols + c][0]
            p2 = corners[(r + 1) * cols + c][0]
            vertical.append(np.linalg.norm(p1 - p2))

    h_mean = np.mean(horizontal)
    v_mean = np.mean(vertical)

    print(f"  Mean horizontal spacing: {h_mean:.3f} px  (n={len(horizontal)})")
    print(f"  Mean vertical spacing:   {v_mean:.3f} px  (n={len(vertical)})")

    average_distance = (h_mean + v_mean) / 2.0
    pixels_per_mm = average_distance / SQUARE_SIZE

    print(f"  Pixels/mm = {pixels_per_mm:.4f}")

    return pixels_per_mm


# ==========================================================
# SEGMENT OBJECT
# ==========================================================

def get_segmentation(image, tag="segmentation"):

    print("\nRunning segmentation...")

    mask, confidence = predictor.predict(image)

    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    _debug_save(f"{tag}_mask.jpg", mask)

    print(f"Segmentation confidence: {confidence:.3f}")

    return mask, confidence


# ==========================================================
# MEASURE OBJECT FROM MASK
# ==========================================================

def measure_object(mask, pixels_per_mm):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError(
            "\nNo object found in segmentation mask.\n"
            "Check model confidence / threshold, or that the "
            "object is actually visible and unoccluded."
        )

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (cx, cy), (w_px, h_px), angle = rect

    width_px, height_px = max(w_px, h_px), min(w_px, h_px)
    width_mm = width_px / pixels_per_mm
    height_mm = height_px / pixels_per_mm

    box = np.intp(cv2.boxPoints(rect))

    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "width_px": width_px,
        "height_px": height_px,
        "box_points": box,
        "contour": largest,
    }


def draw_measurement(image, result, confidence):

    output = image.copy()
    cv2.drawContours(output, [result["box_points"]], 0, (0, 255, 0), 2)

    label = (
        f"W: {result['width_mm']:.1f} mm  "
        f"H: {result['height_mm']:.1f} mm  "
        f"(conf: {confidence:.2f})"
    )

    x, y = result["box_points"][0]
    cv2.putText(
        output, label, (int(x), max(0, int(y) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    return output


# ==========================================================
# SINGLE-IMAGE PIPELINE
# ==========================================================

def run_measurement(image_path, camera_matrix=None, distortion=None, calib_size=None,
                     quiet_diagnostics=False):
    """
    Runs the full pipeline on one image.

    camera_matrix / distortion / calib_size can be passed in (pre-loaded)
    when processing a batch, so calibration is loaded from disk only once
    instead of once per image.
    """

    print("=" * 60)
    print("Object Measurement Pipeline")
    print("=" * 60)
    print("Image:", image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image (corrupt or unsupported): {image_path}"
        )

    if camera_matrix is None or distortion is None or calib_size is None:
        camera_matrix, distortion, calib_size = load_calibration()

    actual_size = (image.shape[1], image.shape[0])  # (w, h)
    if actual_size != calib_size:
        camera_matrix = scale_camera_matrix(camera_matrix, calib_size, actual_size)
    else:
        print("  Image resolution matches calibration resolution exactly — no rescaling needed.")

    undistorted = undistort_image(image, camera_matrix, distortion)

    stem = os.path.splitext(os.path.basename(image_path))[0]

    if not quiet_diagnostics:
        print("\n" + "#" * 60)
        print("# DIAGNOSTIC PASS: detection on RAW (pre-undistortion) image")
        print("#" * 60)
        raw_found, _, _, _ = detect_checkerboard(image, image_tag=f"{stem}_raw")
    else:
        raw_found = None

    print("\n" + "#" * 60)
    print("# MAIN PASS: detection on UNDISTORTED image")
    print("#" * 60)

    und_found, und_corners, und_pattern, und_variant = detect_checkerboard(
        undistorted, image_tag=f"{stem}_undistorted"
    )

    if raw_found is not None:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"  Raw image detection:         {'SUCCESS' if raw_found else 'FAILED'}")
        print(f"  Undistorted image detection: {'SUCCESS' if und_found else 'FAILED'}")
        print("=" * 60)

    if not und_found:
        raise RuntimeError(
            f"\nCheckerboard could not be detected in the undistorted image "
            f"for {image_path}.\nSee debug images in:\n{DEBUG_FOLDER}\n"
        )

    corners, detected_pattern = und_corners, und_pattern

    pixels_per_mm = calculate_pixels_per_mm(corners, detected_pattern)

    mask, confidence = get_segmentation(undistorted, tag=stem)

    result = measure_object(mask, pixels_per_mm)

    print("\n" + "=" * 60)
    print("Measurement Result")
    print("=" * 60)
    print(f"Width  : {result['width_mm']:.2f} mm")
    print(f"Height : {result['height_mm']:.2f} mm")
    print(f"Confidence : {confidence:.3f}")
    print("=" * 60)

    output_image = draw_measurement(undistorted, result, confidence)

    output_path = os.path.join(OUTPUT_FOLDER, f"{stem}_measurement_result.jpg")
    cv2.imwrite(output_path, output_image)

    mask_path = os.path.join(OUTPUT_FOLDER, f"{stem}_segmentation_mask.jpg")
    cv2.imwrite(mask_path, mask)

    print(f"\nSaved: {output_path}")
    print(f"Saved: {mask_path}")

    result["confidence"] = confidence
    result["image_path"] = image_path
    return result


# ==========================================================
# BATCH PIPELINE (folder of images)
# ==========================================================

def get_image_files(folder_path):
    files = []
    for name in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder_path, name))
    return files


def run_batch(folder_path):

    image_files = get_image_files(folder_path)

    print("=" * 60)
    print("BATCH MODE")
    print("=" * 60)
    print(f"Folder: {folder_path}")
    print(f"Found {len(image_files)} image(s)")
    print("=" * 60)

    if not image_files:
        print("ERROR: No images found in folder.")
        return

    # Load calibration once for the whole batch instead of once per image.
    camera_matrix, distortion, calib_size = load_calibration()

    rows = []
    succeeded = 0
    failed = 0

    for idx, image_path in enumerate(image_files):
        print(f"\n\n{'#'*70}")
        print(f"# [{idx+1}/{len(image_files)}] {os.path.basename(image_path)}")
        print(f"{'#'*70}")

        try:
            result = run_measurement(
                image_path,
                camera_matrix=camera_matrix,
                distortion=distortion,
                calib_size=calib_size,
                quiet_diagnostics=True,   # skip the raw-image diagnostic pass in batch mode, for speed
            )
            rows.append({
                "trial_id": idx + 1,
                "image_name": os.path.basename(image_path),
                "pred_width_mm": round(result["width_mm"], 2),
                "pred_height_mm": round(result["height_mm"], 2),
                "confidence": round(result["confidence"], 3),
                "actual_width_mm": "",   # fill in after measuring the physical object
                "actual_height_mm": "",  # fill in after measuring the physical object
                "status": "ok",
            })
            succeeded += 1

        except Exception as e:
            print(f"  ERROR processing {image_path}: {e}")
            rows.append({
                "trial_id": idx + 1,
                "image_name": os.path.basename(image_path),
                "pred_width_mm": "",
                "pred_height_mm": "",
                "confidence": "",
                "actual_width_mm": "",
                "actual_height_mm": "",
                "status": f"FAILED: {e}",
            })
            failed += 1

    fieldnames = [
        "trial_id", "image_name", "pred_width_mm", "pred_height_mm",
        "confidence", "actual_width_mm", "actual_height_mm", "status",
    ]
    with open(TRIALS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Total images:      {len(image_files)}")
    print(f"Succeeded:         {succeeded}")
    print(f"Failed:            {failed}")
    print(f"\nTrials CSV saved to:\n{TRIALS_CSV_PATH}")
    print(
        "\nNEXT STEP: open trials.csv and fill in 'actual_width_mm' / "
        "'actual_height_mm' for each image using a physical ruler/caliper "
        "measurement of that same object placement, then run validate_accuracy.py "
        "on this CSV."
    )
    print("=" * 60)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Measure real-world object dimensions from an image or a folder of images."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to a single input image, OR a folder of images to batch-process."
    )
    args = parser.parse_args()

    if os.path.isdir(args.image_path):
        run_batch(args.image_path)
    else:
        run_measurement(args.image_path)