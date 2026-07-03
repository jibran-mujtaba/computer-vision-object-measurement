import os
import json
import cv2
import numpy as np

# =====================================================
# Project Paths
# =====================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

print("=" * 60)
print("COCO Mask Generation")
print("=" * 60)
print("Project Directory :", PROJECT_DIR)
print("Dataset Directory :", DATASET_DIR)


# =====================================================
# Create Masks
# =====================================================

def create_masks(split):

    dataset_path = os.path.join(DATASET_DIR, split)

    image_folder = os.path.join(dataset_path, "images")
    annotation_file = os.path.join(dataset_path, "annotations.json")
    mask_folder = os.path.join(dataset_path, "masks")

    print(f"\nProcessing split: {split}")
    print("Image Folder :", image_folder)
    print("Annotation   :", annotation_file)

    if not os.path.exists(annotation_file):
        raise FileNotFoundError(
            f"annotations.json not found:\n{annotation_file}"
        )

    os.makedirs(mask_folder, exist_ok=True)

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    annotations_per_image = {}

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        annotations_per_image.setdefault(image_id, []).append(ann)

    print(f"Images      : {len(images)}")
    print(f"Annotations : {len(coco['annotations'])}")

    generated = 0

    for image_id, image_info in images.items():

        width = image_info["width"]
        height = image_info["height"]

        mask = np.zeros((height, width), dtype=np.uint8)

        anns = annotations_per_image.get(image_id, [])

        for ann in anns:

            segmentation = ann.get("segmentation", [])

            # Skip RLE masks
            if not isinstance(segmentation, list):
                continue

            for polygon in segmentation:

                if len(polygon) < 6:
                    continue

                points = np.array(polygon, dtype=np.float32).reshape((-1, 2))
                points = np.round(points).astype(np.int32)

                cv2.fillPoly(mask, [points], 255)

        filename = os.path.splitext(image_info["file_name"])[0]

        output_path = os.path.join(
            mask_folder,
            filename + ".png"
        )

        cv2.imwrite(output_path, mask)

        generated += 1

        print(f"[{generated:03d}/{len(images)}] {filename}.png")

    print(f"\n✓ Generated {generated} masks for '{split}'.")


# =====================================================
# Main
# =====================================================

for split in ["train", "val", "test"]:
    create_masks(split)

print("\n" + "=" * 60)
print("All masks generated successfully.")
print("=" * 60)

