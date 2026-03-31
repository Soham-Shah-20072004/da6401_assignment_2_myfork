"""Segmentation model
"""

import torch
import torch.nn as nn

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """

        super().__init__()

        # encoder part
        self.vgg11_encoder = VGG11Encoder(in_channels)
        # in vgg11 we have forward pass that returns the bottleneck features and the skip maps

        # decoder part. (it takes in the bottleneck output and the skip maps)
        # decoder block i will write as 2 parts,  up sampling and conv part, because there is a requirement of concatenate in between
        self.up1   = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # 1024 after concat
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU()
        )

        self.up2   = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 after concat
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )

        self.up3   = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 after concat
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.up4   = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 after concat
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.up5   = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # no concat at this point
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        # at this point the size of the tensor is [B, 64, 224, 224]
        # we need to convert it to [B, num_classes, 224, 224]

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # what we have here is a 3 channel image of same as input size, where each pixel value is the probability of that pixel belonging to which class.
        # this means each channel is basically is the class - foreground, background, boundary.
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # TODO: Implement forward pass.

        bottleneck, feature_dict = self.vgg11_encoder(x, return_features=True)
        x = self.up1(bottleneck)
        x = torch.cat([x, feature_dict['block4']], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, feature_dict['block3']], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, feature_dict['block2']], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, feature_dict['block1']], dim=1)
        x = self.conv4(x)
        x = self.up5(x)
        x = self.conv5(x)
        x = self.final_conv(x)
        return x
