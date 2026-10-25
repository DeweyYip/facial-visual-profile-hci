"""ResNet50 baseline for multi-label facial attribute classification."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Attributes(nn.Module):
    def __init__(self, num_attributes=24, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = resnet50(weights=weights)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_attributes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)
