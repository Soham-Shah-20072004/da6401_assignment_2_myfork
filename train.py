"""Training entrypoint
"""
import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
import albumentations as A
from albumentations.pytorch import ToTensorV2

CONFIG = {
    "data_root":    "oxford-iiit-pet",
    "batch_size":   32,
    "num_workers":  4,
    "lr":           1e-4,
    "epochs":       30,
    "val_split":    0.2,       # 20% of trainval for validation
    "dropout_p":    0.5,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints",
}


def get_train_transform():
    """Augmentation for training — adds flips, color jitter etc."""
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["bbox_labels"],
        min_visibility=0.3       # drop bbox if mostly cropped out
    ))

def get_val_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["bbox_labels"]
    ))

def get_dataloaders(config):

    full_dataset = OxfordIIITPetDataset(
        root_dir=config["data_root"],
        split="train",
        transform=get_train_transform()
    )

    val_size   = int(len(full_dataset) * config["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_dataset.dataset.transform = get_val_transform()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    return train_loader, val_loader


def save_checkpoint(model, epoch, best_metric, filename):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    path = os.path.join(CONFIG["checkpoint_dir"], filename)
    torch.save({
        "state_dict":  model.state_dict(),
        "epoch":       epoch,
        "best_metric": best_metric,
    }, path)
    print(f"  ✓ Saved checkpoint → {path}")


# ──────────────────────────────────────────────
# Task 1: Train Classifier
# ──────────────────────────────────────────────
def train_classifier(config):
    print("\n" + "="*50)
    print("TASK 1: Training VGG11 Classifier")
    print("="*50)

    wandb.init(project="da6401-assignment2", name="classifier", config=config)

    device = torch.device(config["device"])
    train_loader, val_loader = get_dataloaders(config)

    model     = VGG11Classifier(num_classes=37, dropout_p=config["dropout_p"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0

    for epoch in range(config["epochs"]):
        # ── training phase ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()           # clear old gradients
            logits = model(images)          # forward pass
            loss   = criterion(logits, labels)
            loss.backward()                 # compute gradients
            optimizer.step()               # update weights

            train_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        train_acc  = correct / total
        train_loss = train_loss / len(train_loader)

        # ── validation phase ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():              # no gradients needed for val
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits    = model(images)
                loss      = criterion(logits, labels)
                val_loss += loss.item()
                preds     = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # log to wandb
        wandb.log({
            "classifier/train_loss": train_loss,
            "classifier/train_acc":  train_acc,
            "classifier/val_loss":   val_loss,
            "classifier/val_acc":    val_acc,
            "classifier/lr":         scheduler.get_last_lr()[0],
            "epoch":                 epoch + 1
        })

        # save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, epoch+1, best_acc, "classifier.pth")

    wandb.finish()
    print(f"\nBest Validation Accuracy: {best_acc:.4f}")


# ──────────────────────────────────────────────
# Task 2: Train Localizer
# ──────────────────────────────────────────────
def train_localizer(config):
    print("\n" + "="*50)
    print("TASK 2: Training VGG11 Localizer")
    print("="*50)

    wandb.init(project="da6401-assignment2", name="localizer", config=config)

    device = torch.device(config["device"])
    train_loader, val_loader = get_dataloaders(config)

    model     = VGG11Localizer(dropout_p=config["dropout_p"]).to(device)
    mse_loss  = nn.MSELoss()
    iou_loss  = IoULoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # optionally load pretrained encoder from classifier
    clf_path = os.path.join(config["checkpoint_dir"], "classifier.pth")
    if os.path.exists(clf_path):
        print("  Loading pretrained encoder from classifier.pth...")
        from models.classification import VGG11Classifier
        clf  = VGG11Classifier()
        ckpt = torch.load(clf_path, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        clf.load_state_dict(state)
        model.vgg11_encoder.load_state_dict(clf.vgg11_encoder.state_dict())
        print("  ✓ Encoder weights loaded")

    best_iou = 0.0

    for epoch in range(config["epochs"]):
        # ── training phase ──
        model.train()
        train_loss, train_iou = 0.0, 0.0

        for batch in train_loader:
            images  = batch["image"].to(device)
            targets = batch["bbox"].to(device)   # [B, 4]

            optimizer.zero_grad()
            preds = model(images)                # [B, 4]

            loss_mse = mse_loss(preds, targets)
            loss_iou = iou_loss(preds, targets)
            loss     = loss_mse + loss_iou       # combined loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # track mean IoU as metric (1 - iou_loss)
            train_iou  += (1 - loss_iou.item())

        train_loss = train_loss / len(train_loader)
        train_iou  = train_iou  / len(train_loader)

        # ── validation phase ──
        model.eval()
        val_loss, val_iou = 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                images  = batch["image"].to(device)
                targets = batch["bbox"].to(device)

                preds    = model(images)
                loss_mse = mse_loss(preds, targets)
                loss_iou = iou_loss(preds, targets)
                loss     = loss_mse + loss_iou

                val_loss += loss.item()
                val_iou  += (1 - loss_iou.item())

        val_loss = val_loss / len(val_loader)
        val_iou  = val_iou  / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f}")

        wandb.log({
            "localizer/train_loss": train_loss,
            "localizer/train_iou":  train_iou,
            "localizer/val_loss":   val_loss,
            "localizer/val_iou":    val_iou,
            "localizer/lr":         scheduler.get_last_lr()[0],
            "epoch":                epoch + 1
        })

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, epoch+1, best_iou, "localizer.pth")

    wandb.finish()
    print(f"\nBest Validation IoU: {best_iou:.4f}")


# ──────────────────────────────────────────────
# Task 3: Train UNet
# ──────────────────────────────────────────────
def dice_score(pred_mask, true_mask, num_classes=3, eps=1e-6):
    """Compute mean Dice score across classes."""
    pred_mask = pred_mask.argmax(dim=1)    # [B, H, W] predicted class per pixel
    dice = 0.0
    for c in range(num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        intersection = (pred_c * true_c).sum()
        dice += (2 * intersection + eps) / (pred_c.sum() + true_c.sum() + eps)
    return (dice / num_classes).item()


def train_unet(config):
    print("\n" + "="*50)
    print("TASK 3: Training VGG11 UNet")
    print("="*50)

    wandb.init(project="da6401-assignment2", name="unet", config=config)

    device = torch.device(config["device"])
    train_loader, val_loader = get_dataloaders(config)

    model     = VGG11UNet(num_classes=3, dropout_p=config["dropout_p"]).to(device)
    criterion = nn.CrossEntropyLoss()      # pixel-wise classification loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # load pretrained encoder from classifier
    clf_path = os.path.join(config["checkpoint_dir"], "classifier.pth")
    if os.path.exists(clf_path):
        print("  Loading pretrained encoder from classifier.pth...")
        from models.classification import VGG11Classifier
        clf  = VGG11Classifier()
        ckpt = torch.load(clf_path, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        clf.load_state_dict(state)
        model.vgg11_encoder.load_state_dict(clf.vgg11_encoder.state_dict())
        print("  ✓ Encoder weights loaded")

    best_dice = 0.0

    for epoch in range(config["epochs"]):
        # ── training phase ──
        model.train()
        train_loss, train_dice = 0.0, 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)   # [B, H, W] integer labels

            optimizer.zero_grad()
            logits = model(images)              # [B, 3, H, W]
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(logits.detach(), masks)

        train_loss = train_loss / len(train_loader)
        train_dice = train_dice / len(train_loader)

        # ── validation phase ──
        model.eval()
        val_loss, val_dice = 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks  = batch["mask"].to(device)

                logits    = model(images)
                loss      = criterion(logits, masks)
                val_loss += loss.item()
                val_dice += dice_score(logits, masks)

        val_loss = val_loss / len(val_loader)
        val_dice = val_dice / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f}")

        wandb.log({
            "unet/train_loss": train_loss,
            "unet/train_dice": train_dice,
            "unet/val_loss":   val_loss,
            "unet/val_dice":   val_dice,
            "unet/lr":         scheduler.get_last_lr()[0],
            "epoch":           epoch + 1
        })

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, epoch+1, best_dice, "unet.pth")

    wandb.finish()
    print(f"\nBest Validation Dice: {best_dice:.4f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # train in order — classifier first since localizer + unet
    # both optionally load its pretrained encoder
    train_classifier(CONFIG)
    train_localizer(CONFIG)
    train_unet(CONFIG)