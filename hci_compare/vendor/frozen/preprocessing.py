"""Shared image and landmark preprocessing for every model (baseline and branch)."""
"""This is not Augmentation. This is pre-processing"""

from PIL import Image
import torch
from torchvision.transforms import functional as TF

SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

"""
the ideal image should be 3 × 224 × 224
Mean and std is for normalization
"""

def prepare(image: Image.Image, landmarks: torch.Tensor):

    """
    The input is a raw image of 178 × 218 × 3
    Return normalized image [3,224,224] and five landmark points [10] in [-1,1].
    landmark format: 
    lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
      69          109       106          113   77     142        73         152          108         154
    which is a [10] tensor (being flattened) 
    """
    
    # Unify the shape of the images
    width, height = image.size            # 178 × 218 celebA
    scale = SIZE / max(width, height)     # 224/218
    new_width = round(width * scale)
    new_height = round(height * scale)
    left = (SIZE - new_width) // 2
    top = (SIZE - new_height) // 2


    """
    Create a 224 x 224 balck image and
    Put the resized image into the middle of iit
    """
    resized = image.convert("RGB").resize(
        (new_width, new_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    canvas.paste(resized, (left, top))


    """ 
    Now that we changed the image 
    So the landmarks have to be adjusted as well
    """
    points = landmarks.clone().reshape(5, 2).float()
    # Ajust x and y accordingly
    points[:, 0] = (points[:, 0] + 0.5) * new_width / width - 0.5 + left
    points[:, 1] = (points[:, 1] + 0.5) * new_height / height - 0.5 + top
    points = 2 * points / (SIZE - 1) - 1

    # Normalize the tensor
    image_tensor = TF.normalize(TF.to_tensor(canvas), MEAN, STD)

    # We flatten the landmarks here because it'd be easier for fure concatenation
    # image_tensor is being tensorfied 

    return image_tensor, points.flatten()
