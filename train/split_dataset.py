
import os
import json
import random
import shutil
from collections import defaultdict

# =====================================================
# Configuration
# =====================================================

random.seed(42)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

# =====================================================
# Project Paths
# =====================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATION_FILE = os.path.join(DATASET_DIR, "annotations.json")

print("=" * 60)
print("Dataset Split Script")
print("=" * 60)

print("Project Directory : ", PROJECT_DIR)
print("Dataset Directory : ", DATASET_DIR)
print("Image Directory   : ", IMAGE_DIR)
print("Annotation File   : ", ANNOTATION_FILE)

# =====================================================
# Validate Paths
# =====================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"\nDataset folder not found:\n{DATASET_DIR}")

if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"\nImages folder not found:\n{IMAGE_DIR}")

if not os.path.exists(ANNOTATION_FILE):
    raise FileNotFoundError(f"\nannotations.json not found:\n{ANNOTATION_FILE}")

print("\n✓ Dataset verified.")

# =====================================================
# Load COCO JSON
# =====================================================

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco.get("images", [])
annotations = coco.get("annotations", [])
categories = coco.get("categories", [])

print(f"\nImages       : {len(images)}")
print(f"Annotations  : {len(annotations)}")
print(f"Categories   : {len(categories)}")

if len(images) == 0:
    raise ValueError("No images found in annotations.json")

# =====================================================
# Verify Image Files
# =====================================================

valid_images = []

missing = 0

for img in images:

    path = os.path.join(IMAGE_DIR, img["file_name"])

    if os.path.exists(path):
        valid_images.append(img)
    else:
        missing += 1
        print(f"Missing image: {img['file_name']}")

print(f"\nValid images : {len(valid_images)}")
print(f"Missing      : {missing}")

if len(valid_images) == 0:
    raise RuntimeError("No valid image files found.")

images = valid_images

# =====================================================
# Shuffle
# =====================================================

random.shuffle(images)

total = len(images)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

print("\nDataset Split")
print("-----------------------")
print("Train :", len(train_images))
print("Val   :", len(val_images))
print("Test  :", len(test_images))

# =====================================================
# Create Output Folders
# =====================================================

splits = {
    "train": train_images,
    "val": val_images,
    "test": test_images
}

for split in splits:

    os.makedirs(
        os.path.join(DATASET_DIR, split, "images"),
        exist_ok=True
    )

# =====================================================
# Create annotation lookup
# =====================================================

annotation_lookup = defaultdict(list)

for ann in annotations:
    annotation_lookup[ann["image_id"]].append(ann)

# =====================================================
# Copy Images + Create JSON
# =====================================================

for split_name, split_images in splits.items():

    print(f"\nProcessing {split_name}...")

    split_annotations = []

    copied = 0

    for img in split_images:

        src = os.path.join(
            IMAGE_DIR,
            img["file_name"]
        )

        dst = os.path.join(
            DATASET_DIR,
            split_name,
            "images",
            img["file_name"]
        )

        shutil.copy2(src, dst)

        copied += 1

        split_annotations.extend(
            annotation_lookup[img["id"]]
        )

    split_json = {

        "info": coco.get("info", {}),

        "licenses": coco.get("licenses", []),

        "categories": categories,

        "images": split_images,

        "annotations": split_annotations
    }

    output_json = os.path.join(
        DATASET_DIR,
        split_name,
        "annotations.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_json,
            f,
            indent=4
        )

    print(f"Copied images : {copied}")
    print(f"Annotations   : {len(split_annotations)}")
    print(f"Saved JSON    : {output_json}")

# =====================================================
# Finished
# =====================================================

print("\n" + "=" * 60)
print("Dataset split completed successfully!")
print("=" * 60)

