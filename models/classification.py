"""Classification components
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder  
from .layers import CustomDropout

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        # inherit the machinery of the parent class
        super().__init__()

        # encoder object
        self.vgg11_encoder = VGG11Encoder(in_channels)

        # encoder is set, now we can add different heads depending on the task. 
        # For classification task, we add a classifier head.(it is a sequential object)
        self.classifier_head = nn.Sequential(
            # in case of different size, lets do adaptive pooling to 7 * 7
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(), # for a batch size B, this will create a matrix [B, 7*7*512]
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        # TODO: Implement forward pass.

        # Python automatically calls self.encoder.forward(x) under the hood. 
        # That's what nn.Module.__call__ does — it routes model(x) to model.forward(x).

        x = self.vgg11_encoder(x)
        x = self.classifier_head(x) # this returns the logits only. the softmax is not applied here.
        return x


