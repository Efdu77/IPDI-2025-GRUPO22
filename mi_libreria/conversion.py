
import numpy as np
from PIL import Image

# RGB a YIQ
def rgb_to_yiq(img):
    M = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.275, -0.321],
                  [0.212, -0.523, 0.311]])
    return np.tensordot(img / 255.0, M.T, axes=1)

# YIQ a RGB
def yiq_to_rgb(yiq):
    M_inv = np.array([[1.0, 0.956, 0.621],
                      [1.0, -0.272, -0.647],
                      [1.0, -1.106, 1.703]])
    rgb = np.tensordot(yiq, M_inv.T, axes=1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)