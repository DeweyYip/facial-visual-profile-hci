"""

M3: ResNet18 with a five-landmark geometry branch.

images [B, 3, H, W]
        ↓
     ResNet18
        ↓
image_features [B, 512]
                         \
                          → concat → [B, 544] → Linear → [B, 24]
                         /
landmarks [B, 10]
        ↓
       MLP
        ↓
geometry_features [B, 32]

"""

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

# we make our own unique M3 model that inherits nn.Module, the father of all neural nets
class ResNet18GeometryAttributes(nn.Module):

    """Fuse 512 image features with 32 geometry features."""

    # the rules about our layers is defined inside
    # it asks for the number of final attributes and whether we can use the pretrained ResNet18
    def __init__(self, num_attributes=24, pretrained=True):
        super().__init__()
        
        # if pretrained = true, we use the pretrained weights otherwise we randomly initialize the weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

        # officially create a Resnet18
        # input: [B, 3, H, W]
        # output: [B, 512]
        self.image_encoder = resnet18(weights=weights)

        # get the feature size of ResNet18
        # fc = fully connected layer
        """
        the original fc layer (the last layer) has:
        in_features=512,
        out_features=1000
        """
        image_features = self.image_encoder.fc.in_features # 512

        """
        now we replace fc with nn.Identity, a y(x) = x function
          image
            ↓
          ResNet18
            ↓
          512-dimensional feature
            ↓
         Identity
            ↓
         Still a 512-dimensional feature
         we change it from [B, 1000] to [B, 512]: this is why we can get the geometry fusion work
        """
        self.image_encoder.fc = nn.Identity()

        # geometry branch
        # Five normalized (x, y) points become 32 geometry features. this is a small MLP
        # input: 10 coordinates, namely five (x, y) pairs. [B, 10]
        # output: 32 geometry_features. [B. 32]
        self.geometry_encoder = nn.Sequential(
            nn.Linear(10, 32), # expand 10 neurons into 32 neurons
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # Predict attributes from the concatenated features.
        # classifier has 512 + 32 = 544 neurons as input
        # and finally gives 24 outputs
        self.classifier = nn.Linear(image_features + 32, num_attributes)

    """
    we applies the rules defiend for both
    image_encoder + geometry_encoder + classifier 
    in here
    """
    def forward(
        self,
        images: torch.Tensor,     # input of M3
        landmarks: torch.Tensor,  # another input of M3        notice in baseline resnet18 we only have images as the only input
        geometry_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        # Check the shape of the images and landmarks
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if landmarks.ndim != 2 or landmarks.shape[1] != 10:
            raise ValueError("landmarks must have shape [B, 10]")
        if images.shape[0] != landmarks.shape[0]:
            raise ValueError("images and landmarks must share the batch size")

        # Optional boolean mask: one validity flag per image.
        if geometry_valid is None:
            geometry_valid = torch.ones(
                landmarks.shape[0],
                dtype=torch.bool,
                device=landmarks.device,
            )
        elif (
            geometry_valid.ndim != 1
            or geometry_valid.shape[0] != landmarks.shape[0]
        ):
            raise ValueError("geometry_valid must have shape [B]")
        elif geometry_valid.dtype != torch.bool:
            raise TypeError("geometry_valid must have boolean dtype")

        geometry_valid = geometry_valid.to(device=landmarks.device)
        valid_mask = geometry_valid.unsqueeze(1)

        # Ignore invalid coordinates before they enter the MLP.
        safe_landmarks = torch.where(
            valid_mask, landmarks, torch.zeros_like(landmarks)
        )
        if not torch.isfinite(safe_landmarks).all():
            raise ValueError("Valid landmarks must be finite")

        # image goes into ResNet18
        image_features = self.image_encoder(images)
        """
        EX:
        images.shape = [32, 3, 224, 224]
        image_features.shape = [32, 512] (512 high level visual representation)
        """

        # input: [32, 10]
        # output: [32, 32]
        geometry_features = self.geometry_encoder(safe_landmarks)

        # Mask after the MLP so its biases cannot create missing geometry.
        geometry_features = torch.where(
            valid_mask,
            geometry_features,
            torch.zeros_like(geometry_features),
        )
        
        """
        Fusion:
        image_features.shape = [32, 512]
        geometry_features.shape = [32, 32]
        we follow along dim=1 -> feature dimenson:
        [32, 512] + [32, 32] = [32, 544]
        output: [f1, f2, ..., f512, g1, g2, ..., g32] -> 544 dimensions in total
        """
        combined_features = torch.cat(
            [image_features, geometry_features],
            dim=1,
        )

        # Return logits for BCEWithLogitsLoss
        # [32, 544] -> [32, 24]
        return self.classifier(combined_features)
