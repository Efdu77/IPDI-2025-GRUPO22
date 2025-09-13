import numpy as np

# Calculos basicos de RGB

def suma_rgb_clamp(imgA, imgB):
    return np.clip(imgA.astype(np.int16) + imgB.astype(np.int16), 0, 255).astype(np.uint8)

def suma_rgb_prom(imgA, imgB):
    return ((imgA.astype(np.int16) + imgB.astype(np.int16)) / 2).astype(np.uint8)

def resta_rgb_clamp(imgA, imgB):
    return np.clip(imgA.astype(np.int16) - imgB.astype(np.int16), 0, 255).astype(np.uint8)

def resta_rgb_prom(imgA, imgB):
    return np.clip((imgA.astype(np.int16) - imgB.astype(np.int16)) / 2, 0, 255).astype(np.uint8)

# Calculos basicos de YIQ

def suma_yiq_clamp(yiqA, yiqB):
    Y = np.clip(yiqA[...,0] + yiqB[...,0], 0, 1)
    I = np.clip(yiqA[...,1] + yiqB[...,1], -0.5957, 0.5957)
    Q = np.clip(yiqA[...,2] + yiqB[...,2], -0.5226, 0.5226)
    return np.stack([Y, I, Q], axis=-1)

def resta_yiq_clamp(yiqA, yiqB):
    Y = np.clip(yiqA[...,0] - yiqB[...,0], 0, 1)
    I = np.clip(yiqA[...,1] - yiqB[...,1], -0.5957, 0.5957)
    Q = np.clip(yiqA[...,2] - yiqB[...,2], -0.5226, 0.5226)
    return np.stack([Y, I, Q], axis=-1)

def suma_yiq_prom(yiqA, yiqB):
    Y = np.clip((yiqA[...,0] + yiqB[...,0]) / 2, 0, 1)
    I = np.clip((yiqA[...,1] + yiqB[...,1]) / 2, -0.5957, 0.5957)
    Q = np.clip((yiqA[...,2] + yiqB[...,2]) / 2, -0.5226, 0.5226)
    return np.stack([Y, I, Q], axis=-1)

def resta_yiq_prom(yiqA, yiqB):
    Y = np.clip((yiqA[...,0] - yiqB[...,0]) / 2, 0, 1)
    I = np.clip((yiqA[...,1] - yiqB[...,1]) / 2, -0.5957, 0.5957)
    Q = np.clip((yiqA[...,2] - yiqB[...,2]) / 2, -0.5226, 0.5226)
    return np.stack([Y, I, Q], axis=-1)

# Producto y division (solo RGB y solo un metodo)

def producto(imgA, imgB):
    return ((imgA.astype(np.float32) * imgB.astype(np.float32)) / 255).astype(np.uint8)

def cociente(imgA, imgB):
    return np.clip((imgA.astype(np.float32) / (imgB.astype(np.float32) + 1e-5)) * 255, 0, 255).astype(np.uint8)

# Resta mediante valor absoluto (solo RGB?)

import numpy as np

def resta_abs(imgA, imgB):
    return np.abs(imgA.astype(np.int16) - imgB.astype(np.int16)).astype(np.uint8)
