"""Unified multi-task model
"""

import torch
import torch.nn as nn
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """

        super().__init__()

        # shared backbone encoder. (this is jus inititalization, when we really load the weights we need to know what is the goal, i.e for classifier, different weights will be loaded and for localizer, different weights will be loaded)
        self.encoder = VGG11Encoder(in_channels) # this is Multi-task model's encoder.

        # classification head
        # for now we will load the head from the classifier

        self.classifier = VGG11Classifier(num_breeds, in_channels, dropout_p).classifier_head

        # localization head
        self.localizer = VGG11Localizer(num_breeds, in_channels, dropout_p).regression_head

        # segmentation/ decoder head
        # there is not seperate decoder object in segmentation model. so we will have to initialize all decoder objects one by one. 
        self.segmentation = VGG11UNet(seg_classes, in_channels, dropout_p)

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

        # use this object's own method of loading weights
        # with this in place, when multitask model is initialized, it automatically, initializes with best weights of all three tasks.
        self._load_weights(classifier_path, localizer_path, unet_path)

        pass
    
    def _load_weights(self, classifier_path: str, localizer_path: str, unet_path: str):
        """Load weights for all three tasks.
        Args:
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        clf = VGG11Classifier()
        ckpt = torch.load(classifier_path, map_location="cpu")
        # ckpt is a dictionary with keys:'state_dict', ...
        state = ckpt['state_dict']
        # we only want to load the weights of the classifier head
        # so we need to extract the weights of the classifier head from the state dict
        # the keys of the state dict are of the form 'classifier_head.layer_name'
        # so we need to extract the weights of the classifier head from the state dict
        # and load them into the classifier head of the multitask model
        clf.load_state_dict(state)
        self.encoder.load_state_dict(clf.vgg11_encoder.state_dict())
        self.classifier.load_state_dict(clf.classifier_head.state_dict())

        # loading the localization weights 
        loc = VGG11Localizer()
        ckpt = torch.load(localizer_path, map_location="cpu")
        state = ckpt['state_dict']
        loc.load_state_dict(state)
        self.localizer.load_state_dict(loc.regression_head.state_dict())


        # loading the segmentation decoeder weights
        seg = VGG11UNet()
        ckpt = torch.load(unet_path,map_location="cpu")
        state = ckpt['state_dict']
        seg.load_state_dict(state)
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
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # TODO: Implement forward pass.

        # run encoder ONCE — shared across all tasks
        bottleneck, features = self.encoder(x, return_features=True)

        # --- Task 1: classification ---
        cls_out = self.classifier_head(bottleneck)          # [B, 37]

        # --- Task 2: Localization ---
        loc_out = self.regression_head(bottleneck)          # [B, 4]

        # --- Task 3: Segmentation ---
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
        seg_out = self.final_conv(s)                        # [B, 3, 224, 224]

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out
        }

