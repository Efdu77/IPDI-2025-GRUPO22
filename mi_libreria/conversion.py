
import numpy as np
from PIL import Image

# RGB → YIQ
def rgb_to_yiq(image_path):
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=float) / 255.0
    transform = np.array([
        [0.299,  0.587,  0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523,  0.312]
    ])
    yiq = rgb @ transform.T
    return yiq

# YIQ → RGB
def yiq_to_rgb(yiq):
    transform_inv = np.array([
        [1.0,  0.956,  0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106,  1.703]
    ])
    rgb = yiq @ transform_inv.T
    rgb = np.clip(rgb, 0, 1)  # recortamos valores a [0,1]
    return (rgb * 255).astype(np.uint8)
