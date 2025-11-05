import numpy as np

# Calculos basicos de RGB entre 2 imagenes

def suma_rgb_clamp(imgA, imgB):
    return np.clip(imgA.astype(np.int16) + imgB.astype(np.int16), 0, 255).astype(np.uint8)

def suma_rgb_prom(imgA, imgB):
    return ((imgA.astype(np.int16) + imgB.astype(np.int16)) / 2).astype(np.uint8)

def resta_rgb_clamp(imgA, imgB):
    return np.clip(imgA.astype(np.int16) - imgB.astype(np.int16), 0, 255).astype(np.uint8)

def resta_rgb_prom(imgA, imgB):
    return np.clip((imgA.astype(np.int16) - imgB.astype(np.int16)) / 2, 0, 255).astype(np.uint8)

# Calculos basicos de YIQ entre 2 imagenes

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

def resta_abs(imgA, imgB):
    return np.abs(imgA.astype(np.int16) - imgB.astype(np.int16)).astype(np.uint8)

# Comparar la luminosidad 

def if_lighter(imgA, imgB):
    YA = 0.299*imgA[:,:,0] + 0.587*imgA[:,:,1] + 0.114*imgA[:,:,2]
    YB = 0.299*imgB[:,:,0] + 0.587*imgB[:,:,1] + 0.114*imgB[:,:,2]
    mask = YA > YB
    return np.where(mask[:,:,None], imgA, imgB)

def if_darker(imgA, imgB):
    YA = 0.299*imgA[:,:,0] + 0.587*imgA[:,:,1] + 0.114*imgA[:,:,2]
    YB = 0.299*imgB[:,:,0] + 0.587*imgB[:,:,1] + 0.114*imgB[:,:,2]
    mask = YA < YB
    return np.where(mask[:,:,None], imgA, imgB)

# Operaciones de luminancia

def raiz(Y):
    return np.sqrt(Y)

def cuadrada(Y):
    return np.power(Y, 2)

def lineal_trozos(Y, Ymin=0.2, Ymax=0.8):
    Yp = np.zeros_like(Y)
    Yp[Y < Ymin] = 0
    Yp[Y > Ymax] = 1
    mask = (Y >= Ymin) & (Y <= Ymax)
    Yp[mask] = (Y[mask] - Ymin) / (Ymax - Ymin)
    return Yp

# Función base (convoluciones)

def aplicar_kernel(imagen, kernel):
    if imagen.ndim == 3:
        imagen = imagen.mean(axis=2)
    imagen = imagen.astype(np.float32)
    
    h, w = imagen.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(imagen, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    salida = np.zeros_like(imagen)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            salida[i, j] = np.sum(region * kernel)
    
    salida = np.clip(salida, 0, 255)
    return salida.astype(np.uint8)

# Pasabajos

def pasabajo_plano(size=3):
    return np.ones((size, size)) / (size * size)

def pasabajo_bartlett(size=3):
    if size == 3:
        base = np.array([1, 2, 1])
    elif size == 5:
        base = np.array([1, 2, 3, 2, 1])
    elif size == 7:
        base = np.array([1, 2, 3, 3, 3, 2, 1])
    else:
        raise ValueError("Tamaño no válido para Bartlett (3,5,7).")
    kernel = np.outer(base, base)
    return kernel / np.sum(kernel)

def pasabajo_gaussiano(size=5):
    if size == 5:
        kernel = np.array([
            [1,  4,  7,  4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1,  4,  7,  4, 1]
        ])
    elif size == 7:
        kernel = np.array([
            [0, 0, 1, 2, 1, 0, 0],
            [0, 3, 13, 22, 13, 3, 0],
            [1, 13, 59, 97, 59, 13, 1],
            [2, 22, 97,159, 97, 22, 2],
            [1, 13, 59, 97, 59, 13, 1],
            [0, 3, 13, 22, 13, 3, 0],
            [0, 0, 1, 2, 1, 0, 0]
        ])
    else:
        raise ValueError("Solo tamaños 5x5 y 7x7 válidos.")
    return kernel / np.sum(kernel)

# Laplaciano Y Sobel

def laplaciano_v4():
    return np.array([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]])

def laplaciano_v8():
    return np.array([[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]])

def sobel_orientacion(direccion):
    if direccion == 1:
        return np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])  
    elif direccion == 2:
        return np.array([[0, 1, 2],
                         [-1, 0, 1],
                         [-2, -1, 0]])  
    elif direccion == 3:
        return np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]) 
    elif direccion == 4:
        return np.array([[2, 1, 0],
                         [1, 0, -1],
                         [0, -1, -2]])  
    elif direccion == 5:
        return np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])  
    elif direccion == 6:
        return np.array([[0, -1, -2],
                         [1, 0, -1],
                         [2, 1, 0]])  
    elif direccion == 7:
        return np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])  
    elif direccion == 8:
        return np.array([[-2, -1, 0],
                         [-1, 0, 1],
                         [0, 1, 2]])  
    else:
        raise ValueError("Dirección debe ser un entero entre 1 y 8.")

# Pasaaltos y Pasabanda

def filtro_identidad():
    return np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])

def pasaaltos(fc=0.2, tipo="v4"):
    lap = laplaciano_v4() if tipo == "v4" else laplaciano_v8()
    return filtro_identidad() - fc * lap

def pasabanda(fc=0.2, tipo="v4"):
    lap = laplaciano_v4() if tipo == "v4" else laplaciano_v8()
    return filtro_identidad() + fc * lap

# Morfologia

import numpy as np
from scipy.ndimage import median_filter

def erosion(img):
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)
    h, w = img.shape
    res = np.zeros_like(img)
    padded = np.pad(img, 1, mode='edge')

    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            res[i, j] = np.min(region)
    return res.astype(np.uint8)


def dilatacion(img):
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)
    h, w = img.shape
    res = np.zeros_like(img)
    padded = np.pad(img, 1, mode='edge')

    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            res[i, j] = np.max(region)
    return res.astype(np.uint8)


def apertura(img):
    return dilatacion(erosion(img))


def cierre(img):
    return erosion(dilatacion(img))


def borde_exterior(img):
    return np.clip(dilatacion(img) - img, 0, 255).astype(np.uint8)


def borde_interior(img):
    return np.clip(img - erosion(img), 0, 255).astype(np.uint8)


def gradiente(img):
    return np.clip(dilatacion(img) - erosion(img), 0, 255).astype(np.uint8)


def mediana(img):
    if img.ndim == 3:
        img = img.mean(axis=2)
    return median_filter(img, size=3).astype(np.uint8)

import cv2


# Binarizaciones (?)

def binarizacion_50(img):
    umbral = np.median(img)
    binaria = np.where(img > umbral, 255, 0)
    return binaria.astype(np.uint8)


def binarizacion_dos_modas(img):
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    modas = np.argsort(hist)[-2:]
    umbral = np.mean(modas)
    binaria = np.where(img > umbral, 255, 0)
    return binaria.astype(np.uint8)


def binarizacion_otsu(img):
    img_norm = (img - img.min()) / (img.max() - img.min())
    img_scaled = (img_norm * 99).astype(int)

    hist, _ = np.histogram(img_scaled.ravel(), bins=100, range=(0,100))
    pixel_numb = img.shape[0] * img.shape[1]
    prom_pond = 1 / pixel_numb
    intensity = np.arange(100)

    final_thresh = -1
    final_value = -1

    for x in range(1,100):
        pcb = np.sum(hist[:x])
        pcf = np.sum(hist[x:])
        if pcb == 0 or pcf == 0:
            continue
        wb = pcb * prom_pond
        wf = pcf * prom_pond

        mub = np.sum(intensity[:x] * hist[:x]) / pcb
        muf = np.sum(intensity[x:] * hist[x:]) / pcf

        value = wb * wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = x / 100.0
            final_value = value

    img_bin = np.where(img_norm > final_thresh, 255, 0).astype(np.uint8)
    return img_bin



# Deteccion de bordes

def borde_laplaciano(img):
    lap = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
    return aplicar_kernel(img, lap)

def erosion(img):
    h, w = img.shape
    padded = np.pad(img, 1, mode='edge')
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            out[i, j] = np.min(region)
    return out

def dilatacion(img):
    h, w = img.shape
    padded = np.pad(img, 1, mode='edge')
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            out[i, j] = np.max(region)
    return out

def borde_morfologico(img):
    return np.clip(dilatacion(img) - erosion(img), 0, 255).astype(np.uint8)

def borde_marching_squares(img, nivel=128):
    from skimage import measure
    contornos = measure.find_contours(img, nivel)
    resultado = np.zeros_like(img)
    for c in contornos:
        c = np.round(c).astype(int)
        for y, x in c:
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                resultado[y, x] = 255
    return resultado

# Varita magica

def color_fill(img, x, y, tolerancia=20):
    img_copy = img.copy()
    mask = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)
    cv2.floodFill(img_copy, mask, (x, y), 255,
                  (tolerancia,)*3, (tolerancia,)*3,
                  flags=cv2.FLOODFILL_FIXED_RANGE)
    return img_copy
