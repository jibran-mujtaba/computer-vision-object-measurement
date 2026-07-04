import os
import cv2
import numpy as np

from predictor import Predictor


# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Inner corners
CHECKERBOARD = (7,9)

# Square size in millimetres
SQUARE_SIZE = 20.0


# ==========================================================
# Load Calibration
# ==========================================================

def load_calibration():

    if not os.path.exists(CALIBRATION_FILE):

        raise FileNotFoundError(
            "Calibration file not found.\n"
            "Run calibration/calibrate.py first."
        )

    data = np.load(CALIBRATION_FILE)

    camera_matrix = data["cameraMatrix"]

    distortion = data["distortion"]

    return camera_matrix, distortion


# ==========================================================
# Undistort Image
# ==========================================================

def undistort_image(image, camera_matrix, distortion):

    h, w = image.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(

        camera_matrix,

        distortion,

        (w, h),

        1,

        (w, h)

    )

    undistorted = cv2.undistort(

        image,

        camera_matrix,

        distortion,

        None,

        new_camera_matrix

    )

    return undistorted


# ==========================================================
# Detect Checkerboard
# ==========================================================

# ==========================================================
# Detect Checkerboard
# ==========================================================

def detect_checkerboard(image):
    """
    Robust checkerboard detector.
    - Supports multiple checkerboard sizes.
    - Uses the newer OpenCV SB detector when available.
    - Automatically rescales large images.
    - Saves a debug image showing detected corners.
    """

    print("\n========== CHECKERBOARD DETECTION ==========")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    print(f"Original Size : {w} x {h}")

    # ------------------------------------------------------
    # Resize large images
    # ------------------------------------------------------

    MAX_SIZE = 2000

    scale = min(MAX_SIZE / w, MAX_SIZE / h, 1.0)

    if scale < 1.0:

        gray_small = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        print(f"Resized Image : {gray_small.shape[1]} x {gray_small.shape[0]}")

    else:

        gray_small = gray

    # ------------------------------------------------------
    # Try several checkerboard sizes
    # ------------------------------------------------------

    possible_sizes = [

        (9, 7),
        (7, 9),

        (8, 6),
        (6, 8),

        (8, 7),
        (7, 8),

        (7, 7),
        (8, 8),
        (6, 6)

    ]

    for size in possible_sizes:

        print(f"\nTrying checkerboard size {size}...")

        found = False
        corners = None

        # --------------------------------------------------
        # New OpenCV detector
        # --------------------------------------------------

        if hasattr(cv2, "findChessboardCornersSB"):

            try:

                found, corners = cv2.findChessboardCornersSB(
                    gray_small,
                    size,
                    flags=cv2.CALIB_CB_NORMALIZE_IMAGE
                )

            except Exception:

                found = False

        # --------------------------------------------------
        # Classic detector
        # --------------------------------------------------

        if not found:

            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE |
                cv2.CALIB_CB_FAST_CHECK
            )

            found, corners = cv2.findChessboardCorners(
                gray_small,
                size,
                flags
            )

        if found:

            print(f"\nSUCCESS! Checkerboard detected: {size}")

            # Scale corners back
            if scale < 1:

                corners /= scale

            # Refine
            corners = cv2.cornerSubPix(

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

            debug = image.copy()

            cv2.drawChessboardCorners(
                debug,
                size,
                corners,
                True
            )

            debug_path = os.path.join(
                OUTPUT_FOLDER,
                "checkerboard_detected.jpg"
            )

            cv2.imwrite(debug_path, debug)

            print(f"Debug image saved:\n{debug_path}")

            # Update global checkerboard size

            global CHECKERBOARD

            CHECKERBOARD = size

            return corners

    # ------------------------------------------------------
    # Nothing found
    # ------------------------------------------------------

    fail_path = os.path.join(
        OUTPUT_FOLDER,
        "checkerboard_failed.jpg"
    )

    cv2.imwrite(fail_path, image)

    raise RuntimeError(
        "\nNo checkerboard pattern could be detected.\n\n"
        f"Tried:\n{possible_sizes}\n\n"
        f"Failed image saved to:\n{fail_path}"
    )

    # ------------------------------------------------------
    # Scale back to original image
    # ------------------------------------------------------

    if scale != 1.0:

        corners /= scale

    # ------------------------------------------------------
    # Refine corners
    # ------------------------------------------------------

    corners = cv2.cornerSubPix(
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

    print(f"{len(corners)} corners detected.")

    return corners

# ==========================================================
# Pixels Per Millimetre
# ==========================================================

def calculate_pixels_per_mm(corners):

    horizontal = []

    vertical = []

    rows = CHECKERBOARD[1]
    cols = CHECKERBOARD[0]

    # Horizontal distances
    for r in range(rows):

        for c in range(cols - 1):

            p1 = corners[r * cols + c][0]

            p2 = corners[r * cols + c + 1][0]

            horizontal.append(

                np.linalg.norm(p1 - p2)

            )

    # Vertical distances
    for c in range(cols):

        for r in range(rows - 1):

            p1 = corners[r * cols + c][0]

            p2 = corners[(r + 1) * cols + c][0]

            vertical.append(

                np.linalg.norm(p1 - p2)

            )

    avg_pixels = (

        np.mean(horizontal) +

        np.mean(vertical)

    ) / 2

    pixels_per_mm = avg_pixels / SQUARE_SIZE

    return pixels_per_mm


# ==========================================================
# Predictor
# ==========================================================

predictor = Predictor()


def get_segmentation(image):

    mask, confidence = predictor.predict(image)

    return mask, confidence

# ==========================================================
# Measure Object
# ==========================================================

def measure_object(image_path):

    print("=" * 60)
    print("Object Measurement Started")
    print("=" * 60)

    camera_matrix, distortion = load_calibration()

    print("\n======================================")
    print("Image Path:")
    print(image_path)
    print("======================================")

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    print(f"Image Shape : {image.shape}")
    print(f"Width       : {image.shape[1]}")
    print(f"Height      : {image.shape[0]}")

    original = image.copy()

    print("Undistorting image...")

    image = undistort_image(
        image,
        camera_matrix,
        distortion
    )

    print("Detecting checkerboard...")

    corners = detect_checkerboard(image)

    pixels_per_mm = calculate_pixels_per_mm(corners)

    print(f"Pixels/mm : {pixels_per_mm:.4f}")

    print("Running segmentation...")

    mask, confidence = get_segmentation(image)

    # ------------------------------------------------------
    # Morphological Cleanup
    # ------------------------------------------------------

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # ------------------------------------------------------
    # Find Contours
    # ------------------------------------------------------

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    if len(contours) == 0:

        raise RuntimeError("No object detected.")

    largest = max(

        contours,

        key=cv2.contourArea

    )

    x, y, w, h = cv2.boundingRect(largest)

    width_mm = w / pixels_per_mm

    height_mm = h / pixels_per_mm

    print("\nMeasurements")

    print("---------------------------")

    print(f"Width  : {width_mm:.2f} mm")

    print(f"Height : {height_mm:.2f} mm")

    print(f"Confidence : {confidence*100:.2f}%")

    # ------------------------------------------------------
    # Draw
    # ------------------------------------------------------

    output = image.copy()

    cv2.drawContours(

        output,

        [largest],

        -1,

        (0, 255, 0),

        2

    )

    cv2.rectangle(

        output,

        (x, y),

        (x + w, y + h),

        (255, 0, 0),

        2

    )

    cv2.putText(

        output,

        f"Width : {width_mm:.1f} mm",

        (x, y - 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 0),

        2

    )

    cv2.putText(

        output,

        f"Height : {height_mm:.1f} mm",

        (x, y - 15),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 0),

        2

    )

    cv2.putText(

        output,

        f"Confidence : {confidence*100:.1f}%",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 0, 255),

        2

    )

    filename = os.path.basename(image_path)

    save_path = os.path.join(

        OUTPUT_FOLDER,

        filename

    )

    cv2.imwrite(

        save_path,

        output

    )

    print("\nAnnotated image saved to:")

    print(save_path)

    cv2.imshow("Measurement", output)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return width_mm, height_mm, confidence


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    TEST_FOLDER = os.path.join(

        PROJECT_ROOT,

        "dataset",

        "test",

        "images"

    )

    files = [

        f for f in os.listdir(TEST_FOLDER)

        if f.lower().endswith(

            (".jpg", ".jpeg", ".png")

        )

    ]

    if len(files) == 0:

        print("No test images found.")

        exit()

    print("\nAvailable Test Images\n")

    for i, file in enumerate(files):

        print(f"{i+1}. {file}")

    choice = int(input("\nSelect image number: "))

    image_path = os.path.join(

        TEST_FOLDER,

        files[choice - 1]

    )

    measure_object(image_path)