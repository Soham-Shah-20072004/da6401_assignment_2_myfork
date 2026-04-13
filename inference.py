"""Inference and evaluation
"""

"""Inference and evaluation"""

import os
import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
from train import get_val_transform, dice_score, CONFIG


# ──────────────────────────────────────────────
# Checkpoint loader
# ──────────────────────────────────────────────
def load_model(model, checkpoint_path, device):
    """Load a saved checkpoint into a model."""
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt["state_dict"] 
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Task 1: Evaluate Classifier
# ──────────────────────────────────────────────
def evaluate_classifier(config):
    print("\n" + "="*50)
    print("Evaluating Classifier")
    print("="*50)

    device = torch.device(config["device"])

    # load test dataset
    test_dataset = OxfordIIITPetDataset(
        root_dir=config["data_root"],
        split="test",
        transform=get_val_transform()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    # load model
    model = VGG11Classifier(num_classes=37)
    model = load_model(
        model,
        os.path.join(config["checkpoint_dir"], "classifier.pth"),
        device
    )

    all_preds  = []   # collect all predictions
    all_labels = []   # collect all ground truths

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)  # [B, 3, 224, 224]
            labels = batch["label"]              # keep on CPU for sklearn

            logits = model(images)               # [B, 37]
            preds  = logits.argmax(dim=1).cpu()  # [B] predicted classes

            all_preds.append(preds)
            all_labels.append(labels)

    # concatenate all batches into single tensors
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # compute macro F1 — same metric autograder uses
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy  = (all_preds == all_labels).mean()

    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Test Macro F1:  {macro_f1:.4f}")

    return macro_f1


# ──────────────────────────────────────────────
# Helper: compute IoU between two sets of boxes
# ──────────────────────────────────────────────
def compute_iou_batch(pred_boxes, target_boxes, eps=1e-6):
    """Compute per-sample IoU between predicted and target boxes.
    Args:
        pred_boxes:   [B, 4] in (x_center, y_center, w, h)
        target_boxes: [B, 4] in (x_center, y_center, w, h)
    Returns:
        iou: [B] per-sample IoU values
    """
    # # convert to corners
    # pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    # pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    # pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    # pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    # tgt_x1  = target_boxes[:, 0] - target_boxes[:, 2] / 2
    # tgt_y1  = target_boxes[:, 1] - target_boxes[:, 3] / 2
    # tgt_x2  = target_boxes[:, 0] + target_boxes[:, 2] / 2
    # tgt_y2  = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # inter_x1 = torch.max(pred_x1, tgt_x1)
    # inter_y1 = torch.max(pred_y1, tgt_y1)
    # inter_x2 = torch.min(pred_x2, tgt_x2)
    # inter_y2 = torch.min(pred_y2, tgt_y2)

    # inter_w    = (inter_x2 - inter_x1).clamp(min=0)
    # inter_h    = (inter_y2 - inter_y1).clamp(min=0)
    # inter_area = inter_w * inter_h

    # pred_area  = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    # tgt_area   = (tgt_x2  - tgt_x1)  * (tgt_y2  - tgt_y1)
    # union_area  = pred_area + tgt_area - inter_area + eps

    # return inter_area / union_area   # [B]

    iou_fn = IoULoss(reduction="none")  # per-sample, not averaged
    return 1 - iou_fn(pred_boxes, target_boxes)

# ──────────────────────────────────────────────
# Task 2: Evaluate Localizer
# ──────────────────────────────────────────────
def evaluate_localizer(config):
    print("\n" + "="*50)
    print("Evaluating Localizer")
    print("="*50)

    device = torch.device(config["device"])

    test_dataset = OxfordIIITPetDataset(
        root_dir=config["data_root"],
        split="test",
        transform=get_val_transform()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    model = VGG11Localizer()
    model = load_model(
        model,
        os.path.join(config["checkpoint_dir"], "localizer.pth"),
        device
    )

    all_ious = []

    with torch.no_grad():
        for batch in test_loader:
            images  = batch["image"].to(device)
            targets = batch["bbox"].to(device)    # [B, 4]

            preds = model(images)                 # [B, 4]
            ious  = compute_iou_batch(preds, targets)  # [B]
            all_ious.append(ious.cpu())

    all_ious = torch.cat(all_ious)                # [N] all test samples
    mean_iou = all_ious.mean().item()

    print(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou


# ──────────────────────────────────────────────
# Task 3: Evaluate UNet
# ──────────────────────────────────────────────
def evaluate_unet(config):
    print("\n" + "="*50)
    print("Evaluating UNet")
    print("="*50)

    device = torch.device(config["device"])

    test_dataset = OxfordIIITPetDataset(
        root_dir=config["data_root"],
        split="test",
        transform=get_val_transform()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    model = VGG11UNet(num_classes=3)
    model = load_model(
        model,
        os.path.join(config["checkpoint_dir"], "unet.pth"),
        device
    )

    total_dice         = 0.0
    total_pixel_acc    = 0.0
    num_batches        = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)    # [B, H, W]

            logits     = model(images)            # [B, 3, H, W]
            pred_masks = logits.argmax(dim=1)     # [B, H, W]

            # dice score
            total_dice      += dice_score(logits, masks)

            # pixel accuracy — what fraction of pixels correct
            pixel_acc        = (pred_masks == masks).float().mean().item()
            total_pixel_acc += pixel_acc

            num_batches += 1

    mean_dice      = total_dice      / num_batches
    mean_pixel_acc = total_pixel_acc / num_batches

    print(f"Mean Dice Score:    {mean_dice:.4f}")
    print(f"Mean Pixel Accuracy:{mean_pixel_acc:.4f}")

    return mean_dice


# ──────────────────────────────────────────────
# WandB: log bounding box visualizations
# (for W&B report section 2.5)
# ──────────────────────────────────────────────
def log_bbox_visualizations(config, num_samples=10):
    """Log images with predicted and GT bounding boxes to W&B."""

    device = torch.device(config["device"])

    test_dataset = OxfordIIITPetDataset(
        root_dir=config["data_root"],
        split="test",
        transform=get_val_transform()
    )

    model = VGG11Localizer()
    model = load_model(
        model,
        os.path.join(config["checkpoint_dir"], "localizer.pth"),
        device
    )

    wandb_images = []

    samples_collected = 0
    for i in range(len(test_dataset)):
        if samples_collected >= num_samples:
            break
            
        # Check if the image has a real XML bounding box
        name, _ = test_dataset.samples[i]
        xml_path = os.path.join(test_dataset.xml_dir, name + ".xml")
        if not os.path.exists(xml_path):
            continue  # Skip fake bounding boxes!

        sample = test_dataset[i]
        image  = sample["image"]                  # [3, 224, 224]
        target = sample["bbox"]                   # [4]
        
        samples_collected += 1

        # run prediction
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))  # [1, 4]
            pred = pred.squeeze(0).cpu()                  # [4]

        # compute IoU for this sample
        iou = compute_iou_batch(
            pred.unsqueeze(0),
            target.unsqueeze(0)
        ).item()

        # convert image tensor back to numpy for plotting
        # reverse normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_np = (image * std + mean).clamp(0,1).permute(1,2,0).numpy()

        # plot
        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.imshow(img_np)

        def draw_box(box, color, label):
            cx, cy, w, h = box
            x1, y1 = cx - w/2, cy - h/2
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, label, color=color, fontsize=8)

        draw_box(target.numpy(), "green", "GT")
        draw_box(pred.numpy(),   "red",   f"Pred IoU={iou:.2f}")

        ax.set_title(f"Sample {i} | IoU = {iou:.3f}")
        ax.axis("off")

        wandb_images.append(wandb.Image(fig, caption=f"IoU={iou:.3f}"))
        plt.close(fig)

    wandb.log({"bbox_predictions": wandb_images})


# ──────────────────────────────────────────────
# WandB: log segmentation visualizations
# (for W&B report section 2.6)
# ──────────────────────────────────────────────
def log_seg_visualizations(config, num_samples=5):
    """Log original image, GT mask, predicted mask to W&B."""

    device = torch.device(config["device"])

    test_dataset = OxfordIIITPetDataset(
        root_dir=config["data_root"],
        split="test",
        transform=get_val_transform()
    )

    model = VGG11UNet(num_classes=3)
    model = load_model(
        model,
        os.path.join(config["checkpoint_dir"], "unet.pth"),
        device
    )

    # colors for each class: background, foreground, boundary
    colors = np.array([
        [0,   0,   0  ],   # class 0 — background — black
        [255, 255, 255],   # class 1 — foreground — white
        [128, 128, 128],   # class 2 — boundary   — gray
    ], dtype=np.uint8)

    wandb_images = []

    for i in range(num_samples):
        sample    = test_dataset[i]
        image     = sample["image"]               # [3, 224, 224]
        true_mask = sample["mask"].numpy()        # [224, 224]

        with torch.no_grad():
            logits    = model(image.unsqueeze(0).to(device))  # [1,3,224,224]
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [224,224]

        # reverse normalization for display
        mean   = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std    = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_np = (image * std + mean).clamp(0,1).permute(1,2,0).numpy()

        # convert class indices to RGB colors
        true_rgb = colors[true_mask]              # [224, 224, 3]
        pred_rgb = colors[pred_mask]              # [224, 224, 3]

        # plot all three side by side
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np);       axes[0].set_title("Original");    axes[0].axis("off")
        axes[1].imshow(true_rgb);     axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(pred_rgb);     axes[2].set_title("Prediction");   axes[2].axis("off")

        wandb_images.append(wandb.Image(fig, caption=f"Sample {i}"))
        plt.close(fig)

    wandb.log({"segmentation_predictions": wandb_images})


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":

    wandb.init(project="da6401-assignment2", name="inference", config=CONFIG)

    macro_f1 = evaluate_classifier(CONFIG)
    mean_iou = evaluate_localizer(CONFIG)
    mean_dice = evaluate_unet(CONFIG)

    # log final metrics
    wandb.log({
        "test/classifier_macro_f1": macro_f1,
        "test/localizer_mean_iou":  mean_iou,
        "test/unet_mean_dice":      mean_dice,
    })

    # log visualizations for W&B report
    log_bbox_visualizations(CONFIG, num_samples=10)
    log_seg_visualizations(CONFIG,  num_samples=5)

    wandb.finish()

    print("\n" + "="*50)
    print("Final Results")
    print(f"Classifier Macro F1: {macro_f1:.4f}")
    print(f"Localizer Mean IoU:  {mean_iou:.4f}")
    print(f"UNet Mean Dice:      {mean_dice:.4f}")
    print("="*50)