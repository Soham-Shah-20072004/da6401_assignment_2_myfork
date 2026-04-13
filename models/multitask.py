"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 dropout_p: float = 0.5,
                 classifier_path: str = "checkpoints/classifier.pth",
                 localizer_path: str = "checkpoints/localizer.pth",
                 unet_path: str = "checkpoints/unet.pth"):
        super().__init__()

        # Download checkpoints from Google Drive (required by autograder)
        import gdown
        os.makedirs(os.path.dirname(classifier_path) if os.path.dirname(classifier_path) else "checkpoints", exist_ok=True)
        if not os.path.exists(classifier_path):
            gdown.download(id="1KYjSypxyneIN0-4onf4iKw5OJYfP3vHc", output=classifier_path, quiet=False)
        if not os.path.exists(localizer_path):
            gdown.download(id="1rsdq8jyxtrw1qTP6nh9LRumf10QnOcWE", output=localizer_path, quiet=False)
        if not os.path.exists(unet_path):
            gdown.download(id="1Hx4VUJ5NwC72YhemGfutejElpeBmBrGB", output=unet_path, quiet=False)

        self.encoder = VGG11Encoder(in_channels)

        clf_model = VGG11Classifier(num_breeds, in_channels, dropout_p)
        loc_model = VGG11Localizer(in_channels, dropout_p)
        seg_model = VGG11UNet(seg_classes, in_channels, dropout_p)

        self.classifier = clf_model.classifier_head
        self.localizer = loc_model.regression_head

        self.up1 = seg_model.up1
        self.conv1 = seg_model.conv1
        self.up2 = seg_model.up2
        self.conv2 = seg_model.conv2
        self.up3 = seg_model.up3
        self.conv3 = seg_model.conv3
        self.up4 = seg_model.up4
        self.conv4 = seg_model.conv4
        self.up5 = seg_model.up5
        self.conv5 = seg_model.conv5
        self.final_conv = seg_model.final_conv

        self._load_weights(classifier_path, localizer_path, unet_path)

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        # Step 1: Load classifier — HEAD only (not encoder)
        if os.path.exists(classifier_path):
            clf = VGG11Classifier()
            ckpt = torch.load(classifier_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            clf.load_state_dict(state)
            self.classifier.load_state_dict(clf.classifier_head.state_dict())

        # Step 2: Load localizer — HEAD only (not encoder)
        if os.path.exists(localizer_path):
            loc = VGG11Localizer()
            ckpt = torch.load(localizer_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            loc.load_state_dict(state)
            self.localizer.load_state_dict(loc.regression_head.state_dict())

        # Step 3: Load UNet — decoder AND encoder (encoder loaded LAST so it wins)
        if os.path.exists(unet_path):
            seg = VGG11UNet()
            ckpt = torch.load(unet_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            seg.load_state_dict(state)
            self.encoder.load_state_dict(seg.vgg11_encoder.state_dict())
            self.up1.load_state_dict(seg.up1.state_dict())
            self.conv1.load_state_dict(seg.conv1.state_dict())
            self.up2.load_state_dict(seg.up2.state_dict())
            self.conv2.load_state_dict(seg.conv2.state_dict())
            self.up3.load_state_dict(seg.up3.state_dict())
            self.conv3.load_state_dict(seg.conv3.state_dict())
            self.up4.load_state_dict(seg.up4.state_dict())
            self.conv4.load_state_dict(seg.conv4.state_dict())
            self.up5.load_state_dict(seg.up5.state_dict())
            self.conv5.load_state_dict(seg.conv5.state_dict())
            self.final_conv.load_state_dict(seg.final_conv.state_dict())

    def forward(self, x: torch.Tensor):
        bottleneck, features = self.encoder(x, return_features=True)

        cls_out = self.classifier(bottleneck)
        loc_out = self.localizer(bottleneck)
        loc_out = loc_out * 224  # scale from [0,1] to pixel space

        s = self.up1(bottleneck)
        s = torch.cat([s, features['block4']], dim=1)
        s = self.conv1(s)
        s = self.up2(s)
        s = torch.cat([s, features['block3']], dim=1)
        s = self.conv2(s)
        s = self.up3(s)
        s = torch.cat([s, features['block2']], dim=1)
        s = self.conv3(s)
        s = self.up4(s)
        s = torch.cat([s, features['block1']], dim=1)
        s = self.conv4(s)
        s = self.up5(s)
        s = self.conv5(s)
        seg_out = self.final_conv(s)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }
