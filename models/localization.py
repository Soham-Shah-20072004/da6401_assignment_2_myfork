"""Localization modules
"""

import torch
import torch.nn as nn

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()

        # encoder object
        self.vgg11_encoder = VGG11Encoder(in_channels)

        # encoder is set, now we can add different heads depending on the task. 
        # For localization task, we add a localization head.(it is a sequential object)
        self.localization_head = nn.Sequential(
            # in case of different size, lets do adaptive pooling to 7 * 7
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(), # for a batch size B, this will create a matrix [B, 7*7*512]
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4) # output is 4 values for the 4 coordinates of the bounding box
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        # TODO: Implement forward pass.
        x = self.vgg11_encoder(x)
        x = self.localization_head(x) # pixel value locations, no normalization or activation required
        
        return x
