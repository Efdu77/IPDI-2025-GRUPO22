
import numpy as np

def suma_rgb_clamp(imgA, imgB):
    return np.clip(imgA.astype(np.int16) + imgB.astype(np.int16), 0, 255).astype(np.uint8)

def suma_rgb_prom(imgA, imgB):
    return ((imgA.astype(np.int16) + imgB.astype(np.int16)) / 2).astype(np.uint8)
