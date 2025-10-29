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

# Pasabajo: base (Kernel)

def aplicar_kernel(img, kernel):
    """Aplica convolución 2D con cierre de bordes por repetición."""
    if img.ndim == 3:
        img = img.mean(axis=2)  # convertir a escala de grises
    img = img / 255.0
    h, w = img.shape
    n = kernel.shape[0]
    pad = n // 2
    padded = np.pad(img, pad, mode='edge')
    result = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+n, j:j+n]
            result[i, j] = np.sum(region * kernel)

    return np.clip(result * 255, 0, 255).astype(np.uint8)

# Pasabajo Plano

def pasabajo_plano_3x3(img):
    k = np.ones((3,3)) / 9
    return aplicar_kernel(img, k)

def pasabajo_plano_5x5(img):
    k = np.ones((5,5)) / 25
    return aplicar_kernel(img, k)

def pasabajo_plano_7x7(img):
    k = np.ones((7,7)) / 49
    return aplicar_kernel(img, k)

# Pasabajo Bartlett

def pasabajo_bartlett_3x3(img):
    s = np.array([1,2,1])
    k = np.outer(s, s)
    k = k / k.sum()
    return aplicar_kernel(img, k)

def pasabajo_bartlett_5x5(img):
    s = np.array([1,2,3,2,1])
    k = np.outer(s, s)
    k = k / k.sum()
    return aplicar_kernel(img, k)

def pasabajo_bartlett_7x7(img):
    s = np.array([1,2,3,4,3,2,1])
    k = np.outer(s, s)
    k = k / k.sum()
    return aplicar_kernel(img, k)

# Pasabajo Gaussiano

def pasabajo_gaussiano_5x5(img):
    s = np.array([1,4,6,4,1])
    k = np.outer(s, s)
    k = k / k.sum()
    return aplicar_kernel(img, k)

def pasabajo_gaussiano_7x7(img):
    s = np.array([1,6,15,20,15,6,1])
    k = np.outer(s, s)
    k = k / k.sum()
    return aplicar_kernel(img, k)

# Bordes Laplaciano

def laplaciano_v4(img):
    k = np.array([[0,-1,0],
                  [-1,4,-1],
                  [0,-1,0]])
    return aplicar_kernel(img, k)

def laplaciano_v8(img):
    k = np.array([[-1,-1,-1],
                  [-1,8,-1],
                  [-1,-1,-1]])
    return aplicar_kernel(img, k)

# Bordes Sobel (8 orientaciones)
def sobel_norte(img):
    k = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    return aplicar_kernel(img, k)

def sobel_sur(img):
    k = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    return aplicar_kernel(img, k)

def sobel_este(img):
    k = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    return aplicar_kernel(img, k)

def sobel_oeste(img):
    k = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    return aplicar_kernel(img, k)

def sobel_ne(img):
    k = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    return aplicar_kernel(img, k)

def sobel_no(img):
    k = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    return aplicar_kernel(img, k)

def sobel_se(img):
    k = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
    return aplicar_kernel(img, k)

def sobel_so(img):
    k = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    return aplicar_kernel(img, k)

# Pasabanda

def pasabanda_02(img):
    """Pasabanda (DoG) usando diferencia de Gaussianas con frecuencia de corte 0.2"""
    s1 = np.array([1,4,6,4,1])
    s2 = np.array([1,6,15,20,15,6,1])
    g1 = np.outer(s1, s1) / np.sum(np.outer(s1, s1))
    g2 = np.outer(s2, s2) / np.sum(np.outer(s2, s2))
    k = g2 - g1
    return aplicar_kernel(img, k)

def pasabanda_04(img):
    """Pasabanda (DoG) más amplio (frecuencia de corte 0.4)"""
    s1 = np.array([1,6,15,20,15,6,1])
    s2 = np.array([1,10,45,120,210,252,210,120,45,10,1])  # más ancho
    g1 = np.outer(s1, s1) / np.sum(np.outer(s1, s1))
    g2 = np.outer(s2, s2) / np.sum(np.outer(s2, s2))
    k = g2 - g1[:7,:7]  # recorte al tamaño más chico
    return aplicar_kernel(img, k)

# Pasaaltos

def pasaaltos_02(img):
    """Pasaaltos de frecuencia 0.2"""
    s = np.array([1,4,6,4,1])
    g = np.outer(s, s)
    g = g / g.sum()
    k = np.zeros_like(g)
    k[g.shape[0]//2, g.shape[1]//2] = 1
    k = k - g
    return aplicar_kernel(img, k)

def pasaaltos_04(img):
    """Pasaaltos de frecuencia 0.4"""
    s = np.array([1,6,15,20,15,6,1])
    g = np.outer(s, s)
    g = g / g.sum()
    k = np.zeros_like(g)
    k[g.shape[0]//2, g.shape[1]//2] = 1
    k = k - g
    return aplicar_kernel(img, k)

# Morfologia

import numpy as np
from scipy.ndimage import median_filter

def erosion(img):
    """Erosión: toma el mínimo valor de la vecindad 3x3."""
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
    """Dilatación: toma el máximo valor de la vecindad 3x3."""
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
    """Apertura = Erosión seguida de Dilatación."""
    return dilatacion(erosion(img))


def cierre(img):
    """Cierre = Dilatación seguida de Erosión."""
    return erosion(dilatacion(img))


def borde_exterior(img):
    """Borde exterior = Dilatación - Original."""
    return np.clip(dilatacion(img) - img, 0, 255).astype(np.uint8)


def borde_interior(img):
    """Borde interior = Original - Erosión."""
    return np.clip(img - erosion(img), 0, 255).astype(np.uint8)


def gradiente(img):
    """Gradiente morfológico = Dilatación - Erosión."""
    return np.clip(dilatacion(img) - erosion(img), 0, 255).astype(np.uint8)


def mediana(img):
    """Filtro de mediana (reduce ruido sin borrar bordes)."""
    if img.ndim == 3:
        img = img.mean(axis=2)
    return median_filter(img, size=3).astype(np.uint8)
