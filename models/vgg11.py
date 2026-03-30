"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):

        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # conv layer
            nn.BatchNorm2d(64),                           # normalize
            nn.ReLU(),                                    # activate
            nn.MaxPool2d(kernel_size=2, stride=2)         # halve spatial size
        )

# That's Block 1. When we call `self.block1(x)`, the tensor flows through all 4 layers automatically.

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: Implement forward pass.
        if return_features:
            feature_dict = {}
            x = self.block1(x)
            # save the output and that becomes the input for the next block
            feature_dict['block1'] = x
            x = self.block2(x)
            feature_dict['block2'] = x
            x = self.block3(x)
            feature_dict['block3'] = x
            x = self.block4(x)
            feature_dict['block4'] = x
            x = self.block5(x)
            feature_dict['block5'] = x
            bottleneck = feature_dict['block5']

            return bottleneck, feature_dict

        else:
            x = self.block1(x)
            # save the output and that becomes the input for the next block
            
            x = self.block2(x)
           
            x = self.block3(x)
            
            x = self.block4(x)
            
            bottleneck = self.block5(x)

            return bottleneck

