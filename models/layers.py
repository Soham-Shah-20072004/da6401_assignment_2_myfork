"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        # inherit the machinery of the parent class
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement dropout.

        # this needs to be done only in training mode
        if self.training:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))

            # at this point some neurons would be zero in the mask
            out = x * mask
            # the ones neurons are scaled by 1/(1-p), inverted dropout
            # this is done to address the issue of different average output in training vs the test
            out = out / (1 - self.p)
        else:
            out = x
        return out
        # raise NotImplementedError("Implement CustomDropout.forward")
