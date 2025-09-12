import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

def rgb_to_yiq(img):
    M = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.275, -0.321],
                  [0.212, -0.523, 0.311]])
    return np.tensordot(img/255.0, M.T, axes=1)

def yiq_to_rgb(img_yiq):
    M_inv = np.array([[1.0, 0.956, 0.621],
                      [1.0, -0.272, -0.647],
                      [1.0, -1.105, 1.702]])
    rgb = np.tensordot(img_yiq, M_inv.T, axes=1)
    return np.clip(rgb*255, 0, 255).astype(np.uint8)

class PixelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aritmética de píxeles")
        
        # Botones
        tk.Button(root, text="Cargar Imagen A", command=self.cargar_A).pack()
        tk.Button(root, text="Cargar Imagen B", command=self.cargar_B).pack()
        tk.Button(root, text="Suma RGB (clampeada)", command=self.suma_rgb_clamp).pack()
        tk.Button(root, text="Resta RGB (abs)", command=self.resta_abs).pack()
        tk.Button(root, text="Producto", command=self.producto).pack()
        tk.Button(root, text="If-lighter", command=self.if_lighter).pack()
        tk.Button(root, text="If-darker", command=self.if_darker).pack()
        
        self.label = tk.Label(root)
        self.label.pack()
        
    def cargar_A(self):
        archivo = filedialog.askopenfilename(
            title="Seleccionar Imagen A",
            filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
        )
        if archivo:
            self.imgA = np.array(Image.open(archivo).convert("RGB"))
            # Mostrar imagen cargada
            im = Image.fromarray(self.imgA)
            im.thumbnail((300, 300))
            imtk = ImageTk.PhotoImage(im)
            self.label.config(image=imtk, text="Imagen A cargada", compound="bottom")
            self.label.image = imtk

    
    def cargar_B(self):
        archivo = filedialog.askopenfilename(
            title="Seleccionar Imagen B",
            filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
        )
        if archivo:
            self.imgB = np.array(Image.open(archivo).convert("RGB"))
            # Mostrar imagen cargada
            im = Image.fromarray(self.imgB)
            im.thumbnail((300, 300))
            imtk = ImageTk.PhotoImage(im)
            self.label.config(image=imtk, text="Imagen B cargada", compound="bottom")
            self.label.image = imtk

    
    def mostrar(self, arr):
        im = Image.fromarray(arr)
        imtk = ImageTk.PhotoImage(im)
        self.label.config(image=imtk)
        self.label.image = imtk
    
    def suma_rgb_clamp(self):
        C = np.clip(self.imgA + self.imgB, 0, 255).astype(np.uint8)
        self.mostrar(C)
    
    def resta_abs(self):
        C = np.abs(self.imgA - self.imgB).astype(np.uint8)
        self.mostrar(C)
    
    def producto(self):
        C = ((self.imgA.astype(np.float32) * self.imgB.astype(np.float32))/255).astype(np.uint8)
        self.mostrar(C)
    
    def if_lighter(self):
        YA = 0.299*self.imgA[:,:,0] + 0.587*self.imgA[:,:,1] + 0.114*self.imgA[:,:,2]
        YB = 0.299*self.imgB[:,:,0] + 0.587*self.imgB[:,:,1] + 0.114*self.imgB[:,:,2]
        mask = YA > YB
        C = np.where(mask[:,:,None], self.imgA, self.imgB)
        self.mostrar(C)
    
    def if_darker(self):
        YA = 0.299*self.imgA[:,:,0] + 0.587*self.imgA[:,:,1] + 0.114*self.imgA[:,:,2]
        YB = 0.299*self.imgB[:,:,0] + 0.587*self.imgB[:,:,1] + 0.114*self.imgB[:,:,2]
        mask = YA < YB
        C = np.where(mask[:,:,None], self.imgA, self.imgB)
        self.mostrar(C)

if __name__ == "__main__":
    root = tk.Tk()
    app = PixelApp(root)
    root.mainloop()
