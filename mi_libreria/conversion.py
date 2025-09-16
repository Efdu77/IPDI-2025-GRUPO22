import numpy as np

def rgb_to_yiq(img):
    # Si está en [0,255], normalizar
    if img.max() > 1:
        img = img / 255.0

    M = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.274, -0.322],
                  [0.211, -0.523, 0.312]])
    yiq = np.dot(img, M.T)

    # Clampeos según PDF
    yiq[:,:,0] = np.clip(yiq[:,:,0], 0, 1)           # Y
    yiq[:,:,1] = np.clip(yiq[:,:,1], -0.5957, 0.5957) # I
    yiq[:,:,2] = np.clip(yiq[:,:,2], -0.5226, 0.5226) # Q

    return yiq

def yiq_to_rgb(yiq):
    M_inv = np.array([[1.0, 0.956, 0.621],
                      [1.0, -0.272, -0.647],
                      [1.0, -1.106, 1.703]])
    rgb = np.dot(yiq, M_inv.T)

    # Recortar a [0,1]
    rgb = np.clip(rgb, 0, 1)

    # Escalar a [0,255]
    return (rgb * 255).astype(np.uint8)
