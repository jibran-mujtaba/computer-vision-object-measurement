import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from unet import UNet


# ============================================================
# Configuration
# ============================================================

# ============================================================
# Project Paths
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATASET_DIR, "train", "masks")

VAL_IMAGE_DIR = os.path.join(DATASET_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(DATASET_DIR, "val", "masks")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
CSV_LOG = os.path.join(OUTPUT_DIR, "training_log.csv")
LOSS_CURVE = os.path.join(OUTPUT_DIR, "loss_curve.png")

print("=" * 60)
print("Project Directory :", PROJECT_DIR)
print("Dataset Directory :", DATASET_DIR)
print("Train Images      :", TRAIN_IMAGE_DIR)
print("Train Masks       :", TRAIN_MASK_DIR)
print("Validation Images :", VAL_IMAGE_DIR)
print("Validation Masks  :", VAL_MASK_DIR)
print("=" * 60)

for path in [
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required directory not found:\n{path}")

NUM_EPOCHS = 50

BATCH_SIZE = 4

LEARNING_RATE = 1e-4

IMAGE_SIZE = 640

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

os.makedirs("checkpoints", exist_ok=True)


# ============================================================
# Dataset
# ============================================================

class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir

        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))

        self.transform = transforms.Compose([

            transforms.ToTensor()

        ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(

            self.image_dir,

            image_name

        )

        mask_name = os.path.splitext(image_name)[0] + ".png"

        mask_path = os.path.join(

            self.mask_dir,

            mask_name

        )

        image = cv2.imread(image_path)

        image = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB

        )

        mask = cv2.imread(

            mask_path,

            cv2.IMREAD_GRAYSCALE

        )

        image = self.transform(image)

        mask = torch.tensor(

            mask,

            dtype=torch.float32

        )

        mask = mask.unsqueeze(0)

        mask = mask / 255.0

        return image, mask


# ============================================================
# DataLoaders
# ============================================================

train_dataset = SegmentationDataset(

    TRAIN_IMAGE_DIR,

    TRAIN_MASK_DIR

)

val_dataset = SegmentationDataset(

    VAL_IMAGE_DIR,

    VAL_MASK_DIR

)

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self,pred,target):

        pred=torch.sigmoid(pred)

        pred=pred.contiguous()

        target=target.contiguous()

        intersection=(pred*target).sum(dim=(2,3))

        union=pred.sum(dim=(2,3))+target.sum(dim=(2,3))

        dice=(2*intersection+1)/(union+1)

        loss=1-dice

        return loss.mean()


bce_loss=nn.BCEWithLogitsLoss()

dice_loss=DiceLoss()


def total_loss(pred,target):

    return bce_loss(pred,target)+dice_loss(pred,target)

# ============================================================
# Evaluation Metrics
# ============================================================

def calculate_iou(pred, target, threshold=0.5):

    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum((2, 3))
    union = pred.sum((2, 3)) + target.sum((2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean().item()


def calculate_dice(pred, target, threshold=0.5):

    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum((2, 3))

    dice = (2 * intersection + 1e-6) / (
        pred.sum((2, 3)) + target.sum((2, 3)) + 1e-6
    )

    return dice.mean().item()


def calculate_precision(pred, target, threshold=0.5):

    pred = torch.sigmoid(pred)
    pred = (pred > threshold).cpu().numpy().flatten()

    target = target.cpu().numpy().flatten()

    return precision_score(
        target,
        pred,
        zero_division=0
    )


def calculate_recall(pred, target, threshold=0.5):

    pred = torch.sigmoid(pred)
    pred = (pred > threshold).cpu().numpy().flatten()

    target = target.cpu().numpy().flatten()

    return recall_score(
        target,
        pred,
        zero_division=0
    )


# ============================================================
# Model
# ============================================================

model = UNet(
    n_channels=3,
    n_classes=1
).to(DEVICE)


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=5

)


scaler = torch.cuda.amp.GradScaler(
    enabled=torch.cuda.is_available()
)


# ============================================================
# Training Function
# ============================================================

def train_one_epoch(epoch):

    model.train()

    running_loss = 0

    running_iou = 0

    running_dice = 0

    progress = tqdm(

        train_loader,

        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",

        leave=False

    )

    for images, masks in progress:

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

            loss = total_loss(
                outputs,
                masks
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        running_iou += calculate_iou(
            outputs,
            masks
        )

        running_dice += calculate_dice(
            outputs,
            masks
        )

        progress.set_postfix(

            Loss=f"{loss.item():.4f}"

        )

    epoch_loss = running_loss / len(train_loader)

    epoch_iou = running_iou / len(train_loader)

    epoch_dice = running_dice / len(train_loader)

    return epoch_loss, epoch_iou, epoch_dice


# ============================================================
# Validation Function
# ============================================================

def validate():

    model.eval()

    running_loss = 0

    running_iou = 0

    running_dice = 0

    running_precision = 0

    running_recall = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = total_loss(
                outputs,
                masks
            )

            running_loss += loss.item()

            running_iou += calculate_iou(
                outputs,
                masks
            )

            running_dice += calculate_dice(
                outputs,
                masks
            )

            running_precision += calculate_precision(
                outputs,
                masks
            )

            running_recall += calculate_recall(
                outputs,
                masks
            )

    val_loss = running_loss / len(val_loader)

    val_iou = running_iou / len(val_loader)

    val_dice = running_dice / len(val_loader)

    val_precision = running_precision / len(val_loader)

    val_recall = running_recall / len(val_loader)

    return (

        val_loss,

        val_iou,

        val_dice,

        val_precision,

        val_recall

    )

# ============================================================
# Main Training Loop
# ============================================================

train_losses = []
val_losses = []

train_ious = []
val_ious = []

train_dices = []
val_dices = []

best_iou = 0.0

history = []


def save_history():

    df = pd.DataFrame(history)

    df.to_csv(CSV_LOG, index=False)


def plot_curves():

    plt.figure(figsize=(12,5))

    # ---------------- Loss ----------------
    plt.subplot(1,2,1)

    plt.plot(train_losses,label="Train Loss")

    plt.plot(val_losses,label="Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Loss Curve")

    plt.grid(True)

    plt.legend()

    # ---------------- IoU ----------------

    plt.subplot(1,2,2)

    plt.plot(train_ious,label="Train IoU")

    plt.plot(val_ious,label="Validation IoU")

    plt.xlabel("Epoch")

    plt.ylabel("IoU")

    plt.title("IoU Curve")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(LOSS_CURVE,dpi=300)

    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("="*70)

    print("Object Measurement System")

    print("U-Net Training")

    print("="*70)

    print(f"Device          : {DEVICE}")

    print(f"Epochs          : {NUM_EPOCHS}")

    print(f"Batch Size      : {BATCH_SIZE}")

    print(f"Learning Rate   : {LEARNING_RATE}")

    print(f"Training Images : {len(train_dataset)}")

    print(f"Validation Img  : {len(val_dataset)}")

    print("="*70)

    for epoch in range(NUM_EPOCHS):

        train_loss,train_iou,train_dice = train_one_epoch(epoch)

        (
            val_loss,
            val_iou,
            val_dice,
            val_precision,
            val_recall
        ) = validate()

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_ious.append(train_iou)
        val_ious.append(val_iou)

        train_dices.append(train_dice)
        val_dices.append(val_dice)

        history.append({

            "Epoch":epoch+1,

            "Train Loss":train_loss,

            "Validation Loss":val_loss,

            "Train IoU":train_iou,

            "Validation IoU":val_iou,

            "Train Dice":train_dice,

            "Validation Dice":val_dice,

            "Precision":val_precision,

            "Recall":val_recall,

            "Learning Rate":optimizer.param_groups[0]["lr"]

        })

        save_history()

        plot_curves()

        if val_iou > best_iou:

            best_iou = val_iou

            torch.save(

                {

                    "epoch":epoch+1,

                    "model_state_dict":model.state_dict(),

                    "optimizer_state_dict":optimizer.state_dict(),

                    "best_iou":best_iou

                },

                MODEL_PATH

            )

            print("\n✓ Best model saved.")

        print(

            f"\nEpoch {epoch+1}/{NUM_EPOCHS}"

        )

        print(

            f"Train Loss : {train_loss:.4f}"

        )

        print(

            f"Val Loss   : {val_loss:.4f}"

        )

        print(

            f"Train IoU  : {train_iou:.4f}"

        )

        print(

            f"Val IoU    : {val_iou:.4f}"

        )

        print(

            f"Train Dice : {train_dice:.4f}"

        )

        print(

            f"Val Dice   : {val_dice:.4f}"

        )

        print(

            f"Precision  : {val_precision:.4f}"

        )

        print(

            f"Recall     : {val_recall:.4f}"

        )

        print("-"*70)

    print("\n")

    print("="*70)

    print("Training Finished Successfully!")

    print("="*70)

    print(f"Best Validation IoU : {best_iou:.4f}")

    print(f"Best Model Saved    : {MODEL_PATH}")

    print(f"Training Log Saved  : {CSV_LOG}")

    print(f"Loss Curve Saved    : {LOSS_CURVE}")

    print("="*70)