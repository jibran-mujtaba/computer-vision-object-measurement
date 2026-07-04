import os
import sys
import cv2
import torch
import numpy as np

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_ROOT)

from models.unet import UNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "checkpoints",
    "best_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 640

THRESHOLD = 0.50


# --------------------------------------------------
# Predictor Class
# --------------------------------------------------

class Predictor:

    def __init__(self):

        print("Loading trained U-Net model...")

        self.model = UNet()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(DEVICE)

        self.model.eval()

        print("Model loaded successfully.\n")


    # --------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------

    def preprocess(self, image):

        original_height, original_width = image.shape[:2]

        resized = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        rgb = rgb.astype(np.float32) / 255.0

        tensor = torch.from_numpy(rgb)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(DEVICE)

        return tensor, original_width, original_height


    # --------------------------------------------------
    # Predict mask
    # --------------------------------------------------

    def predict(self, image):

        tensor, original_width, original_height = self.preprocess(image)

        with torch.no_grad():

            output = self.model(tensor)

            probability = torch.sigmoid(output)

        probability = probability.squeeze().cpu().numpy()

        confidence = 0.0

        object_pixels = probability[
            probability > THRESHOLD
        ]

        if len(object_pixels) > 0:

            confidence = float(object_pixels.mean())

        binary_mask = (probability > THRESHOLD).astype(np.uint8)

        binary_mask *= 255

        binary_mask = cv2.resize(

            binary_mask,

            (original_width, original_height),

            interpolation=cv2.INTER_NEAREST

        )

        return binary_mask, confidence


# --------------------------------------------------
# Standalone Test
# --------------------------------------------------

if __name__ == "__main__":

    TEST_IMAGE = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "test",
        "images"
    )

    images = [

        f for f in os.listdir(TEST_IMAGE)

        if f.lower().endswith((".jpg", ".jpeg", ".png"))

    ]

    if len(images) == 0:

        print("No images found.")

        exit()

    image_path = os.path.join(
        TEST_IMAGE,
        images[0]
    )

    image = cv2.imread(image_path)

    predictor = Predictor()

    mask, confidence = predictor.predict(image)

    print(f"Confidence : {confidence:.3f}")

    cv2.imshow("Original", image)

    cv2.imshow("Predicted Mask", mask)

    cv2.waitKey(0)

    cv2.destroyAllWindows()