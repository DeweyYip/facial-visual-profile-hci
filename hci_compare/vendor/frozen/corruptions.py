"""
Draft v1 corruptions on the shared 224x224 RGB canvas, before normalization.

CelebA original picture
   ↓
resize + padding
   ↓
224×224 RGB canvas (by benchmark_preprocessing.py)
   ↓
corruptions.py
   ↓
corrupted 224×224 RGB canvas
   ↓
Normalize (by preproccessing.py)
   ↓
model

we will normalize the result later
"""

import hashlib
import io
import math

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# the current version 
VERSION = "day14_draft_v1"

# intensity design
# 0: clean,  4: the strongest
# value = LEVELS["rotation"][3 - 1] 
LEVELS = {
    "blur": (1.0, 2.0, 3.0, 4.0),
    "brightness": (0.8, 0.6, 0.4, 0.2),
    "rotation": (5.0, 10.0, 15.0, 20.0),
    "occlusion": (0.10, 0.20, 0.30, 0.40),
    "jpeg": (75, 50, 25, 10),
}


# generate a "fingerprint" to the current image
def rgb_sha256(image):
    """Hash final decoded pixels with mode and dimensions, not JPEG file bytes."""
    header = f"{image.mode}:{image.width}:{image.height}:".encode("ascii")
    return hashlib.sha256(header + image.tobytes()).hexdigest()



"""
input:
canvas       → 224×224 RGB image
kind         → which corruption to use
severity     → 0~4
sample_id    → e.g. "000001.jpg"
seed         → reproducibility
gt_pixels    → 5 GT landmark coordinaton
"""
def apply_corruption(canvas, kind, severity, *, sample_id, seed=0,
                     gt_pixels=None):
    """Return (corrupted final RGB image, transformed GT pixels or None, metadata).

    severity=0 is clean; other levels are 1..4. Each call starts from clean.
    gt_pixels, when supplied, must be finite [5,2] final-canvas pixel coordinates.
    Seeds fix rotation direction across severities and model/loader order.
    Draft implementation choices must be reviewed on validation before freeze.
    """

    """
    CHECK:
    1. only does benchmark canvas
    2. only odes kinds specified above
    3. severity must be in 0, 1, 2, 3, 4
    """
    if canvas.mode != "RGB" or canvas.size != (224, 224):
        raise ValueError("Expected unnormalized RGB canvas of size 224x224")
    if kind not in LEVELS:
        raise ValueError(f"Unknown corruption: {kind}")
    if type(severity) is not int or severity not in range(5):
        raise ValueError("severity must be integer 0..4")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty stable filename")
    
    # Landmork check
    points = None
    if gt_pixels is not None:
        points = np.asarray(gt_pixels, dtype=np.float64).copy()
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise ValueError("GT pixels must be finite [5,2]")
        
    # every severity starts with a clean canvas. There's no accumullation
    output = canvas.copy()

    params = {}
    # only starts when severity is not 0
    if severity:
        # severity value
        value = LEVELS[kind][severity - 1]

        # Gaussian blur and brightness
        if kind == "blur":
            output = canvas.filter(ImageFilter.GaussianBlur(radius=value))
            params = {"sigma_pixels": value, "implementation": "Pillow.GaussianBlur"}
        elif kind == "brightness":
            output = ImageEnhance.Brightness(canvas).enhance(value)
            params = {"factor": value}
        elif kind == "rotation":
            # Deliberately excludes severity so the sign is shared across levels.
            payload = f"{VERSION}|{seed}|{sample_id}|rotation".encode("utf-8")
            digest = hashlib.sha256(payload).digest()
            direction = 1 if digest[0] & 1 else -1
            angle = direction * value                   # this is where we apply the value
            # this is where we acctually do rotation
            output = canvas.rotate(
                angle, resample=Image.Resampling.BILINEAR, expand=False,
                center=(112.0, 112.0), fillcolor=(0, 0, 0),
            )
            # Pillow centers use pixel-edge coordinates; GT uses pixel centers.
            # x_edge=x_pixel+0.5, so edge center 112 means pixel center 111.5.
            theta = math.radians(angle)
            cos, sin = math.cos(theta), math.sin(theta)
            forward = np.array([[cos, sin], [-sin, cos]], dtype=np.float64)
            center = np.array([111.5, 111.5])
            offset = center - forward @ center
            # this is the only kind that changes the landmark points
            if points is not None:
                points = points @ forward.T + offset
            params = {
                "angle_degrees_ccw": angle,
                "pillow_center_edge_coordinates": [112.0, 112.0],
                "gt_center_pixel_coordinates": center.tolist(),
                "forward_matrix_pixel_coordinates": np.column_stack(
                    [forward, offset]).tolist(),
                "resampling": "BILINEAR", "expand": False,
                "fill_rgb": [0, 0, 0],
                "direction_sha256": digest.hex(),
            }
        elif kind == "occlusion":
            # adding a black box into the image
            side = int(math.floor(224 * value + 0.5))
            left, top = 112 - side // 2, 168 - side // 2
            box = (left, top, left + side, top + side)
            output.paste((0, 0, 0), box)
            params = {
                "width_fraction": value, "side_pixels": side,
                "nominal_center_edge_coordinates": [112, 168],
                "box_xyxy_exclusive": list(box), "fill_rgb": [0, 0, 0],
            }
        elif kind == "jpeg":
            buffer = io.BytesIO()
            canvas.save(buffer, format="JPEG", quality=value,
                        subsampling=2, optimize=False, progressive=False)
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                output = encoded.convert("RGB").copy()
            params = {"quality": value, "subsampling": 2,
                      "optimize": False, "progressive": False}
            #meta data for the details of this corruption
    metadata = {
        "implementation_version": VERSION, "sample_id": sample_id,
        "seed": seed, "kind": kind, "severity": severity,
        "parameters": params, "input_rgb_sha256": rgb_sha256(canvas),
        "output_rgb_sha256": rgb_sha256(output),
    }
    # To check if there's any landmark that goes outside of the canvas
    if points is not None:
        metadata["gt_in_frame"] = (
            (points >= 0).all(axis=1) & (points <= 223).all(axis=1)
        ).tolist()
    return output, points, metadata
