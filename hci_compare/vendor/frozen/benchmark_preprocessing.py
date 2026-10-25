"""
Split the preprocessing step in preprocessing.py into separate parts
so we can insert corruption before normalization.

we don't change the `prepare() funtion` 
because we don't wanna break the baseline 
"""
from PIL import Image
import torch
from torchvision.transforms import functional as TF

# keep the parameters the same with preprocessing
from .preprocessing import SIZE, MEAN, STD

"""
    Transform any original CelebA image into a unified 224×224 RGB canvas
    while mapping the ground truth landmarks to this new coordinate system.
"""
def prepare_canvas(image, landmarks):
    width, height = image.size
    scale = SIZE / max(width, height)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (SIZE - new_width) // 2, (SIZE - new_height) // 2
    resized = image.convert("RGB").resize(
        (new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    canvas.paste(resized, (left, top))
    points = landmarks.clone().reshape(5, 2).float()
    points[:, 0] = (points[:, 0] + 0.5) * new_width / width - 0.5 + left
    points[:, 1] = (points[:, 1] + 0.5) * new_height / height - 0.5 + top
    return canvas, points


# Normalizaton
"""
can only process the standard 224×224 RGB canvas 
the result is a tensor
"""
def normalize_canvas(canvas):
    if canvas.mode != "RGB" or canvas.size != (SIZE, SIZE):
        raise ValueError("Expected final unnormalized RGB 224x224 canvas")
    return TF.normalize(TF.to_tensor(canvas), MEAN, STD)

"""
Normalize the landmark points
0 -> -1
223 -> 1
"""
def normalize_points(pixel_points):
    """For transformed GT or newly detected final-canvas points; no resizing."""
    points = torch.as_tensor(pixel_points, dtype=torch.float32)
    if points.shape not in ((5, 2), (10,)) or not torch.isfinite(points).all():
        raise ValueError("Expected finite five-point pixel coordinates")
    return (2 * points.reshape(5, 2) / (SIZE - 1) - 1).flatten()


"""
The overall flow:
                 preprocessing.py
                 ┌───────────────┐
                 │ SIZE          │
                 │ MEAN          │
                 │ STD           │
                 └───────┬───────┘
                         │
                         ▼
        original CelebA image
                │
                ▼
        prepare_canvas()
                │
                ├──→ 224×224 RGB canvas
                │
                └──→ GT landmarks in pixel coordinates
                                │
                                ▼
                        corruptions.py
                                │
                    ┌──────────┴──────────┐
                    │                     │
            image corruption       landmark transform
                    │                     │
                    └──────────┬──────────┘
                                ▼
                        corrupted canvas
                                │
                                ▼
                        normalize_canvas()
                                │
                                ▼
                        [3,224,224]
                                │
                                ▼
                            Model
"""