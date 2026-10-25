"""ResNet18 baseline for multi-label facial attribute classification."""
"""We import the model ResNet18 from torchvision"""
"""All the layers and blocks have been prepared by torchvision"""

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

# ResNet18Attributes is our own model
class ResNet18Attributes(nn.Module):

    # We have 24 attributes. And we will use the pretrained values of ResNet18
    def __init__(self, num_attributes=24, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

        # We directly borrow al the settings from usenet18
        self.network = resnet18(weights=weights)

        """
        After ResNet 18 we get a result of [32, 512] (32 is the batch size)
        this [32, 512] eventually goes to the Linear layer
        """
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_attributes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # It returns the result in a logit form
        return self.network(images)
