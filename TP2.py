from mi_libreria import rgb_to_yiq, yiq_to_rgb, suma_rgb_prom, suma_rgb_clamp

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# ---- Conversiones RGB <-> YIQ ----


def clamp_yiq(yiq):
    Y = np.clip(yiq[...,0], 0, 1)
    I = np.clip(yiq[...,1], -0.5957, 0.5957)
    Q = np.clip(yiq[...,2], -0.5226, 0.5226)
    return np.stack([Y,I,Q], axis=-1)

# ---- App ----
class PixelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aritmética de Píxeles - según PDF")

        # ---- Frame superior: imágenes ----
        frame_imgs = tk.Frame(root)
        frame_imgs.pack(pady=10)

        # Etiquetas de las 3 imágenes
        tk.Label(frame_imgs, text="Imagen A").grid(row=0, column=0)
        tk.Label(frame_imgs, text="Resultado").grid(row=0, column=1)
        tk.Label(frame_imgs, text="Imagen B").grid(row=0, column=2)

        # Canvases
        self.canvas_A = tk.Label(frame_imgs)
        self.canvas_A.grid(row=1, column=0, padx=10)
        self.canvas_res = tk.Label(frame_imgs)
        self.canvas_res.grid(row=1, column=1, padx=10)
        self.canvas_B = tk.Label(frame_imgs)
        self.canvas_B.grid(row=1, column=2, padx=10)

        # ---- Frame inferior: botones ----
        frame_btns = tk.Frame(root)
        frame_btns.pack(pady=10)

        # Botones cargar
        tk.Button(frame_btns, text="Cargar Imagen A", command=self.cargar_A).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(frame_btns, text="Cargar Imagen B", command=self.cargar_B).grid(row=0, column=1, padx=5, pady=5)

        # Operaciones en RGB
        tk.Button(frame_btns, text="Suma RGB (clampeada)", command=self.suma_rgb_clamp).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(frame_btns, text="Resta RGB (clampeada)", command=self.resta_rgb_clamp).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(frame_btns, text="Suma RGB (promediada)", command=self.suma_rgb_prom).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(frame_btns, text="Resta RGB (promediada)", command=self.resta_rgb_prom).grid(row=2, column=1, padx=5, pady=5)

        # Operaciones en YIQ
        tk.Button(frame_btns, text="Suma YIQ", command=self.suma_yiq).grid(row=3, column=0, padx=5, pady=5)
        tk.Button(frame_btns, text="Resta YIQ", command=self.resta_yiq).grid(row=3, column=1, padx=5, pady=5)

        # Producto / Cociente
        tk.Button(frame_btns, text="Producto", command=self.producto).grid(row=4, column=0, padx=5, pady=5)
        tk.Button(frame_btns, text="Cociente", command=self.cociente).grid(row=4, column=1, padx=5, pady=5)

        # Otros
        tk.Button(frame_btns, text="Resta abs", command=self.resta_abs).grid(row=5, column=0, padx=5, pady=5)
        tk.Button(frame_btns, text="If-lighter", command=self.if_lighter).grid(row=6, column=0, padx=5, pady=5)
        tk.Button(frame_btns, text="If-darker", command=self.if_darker).grid(row=6, column=1, padx=5, pady=5)

        # Variables para imágenes cargadas
        self.imgA = None
        self.imgB = None

    # ---- Funciones auxiliares ----
    def cargar_A(self):
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png;*.bmp")])
        if archivo:
            self.imgA = np.array(Image.open(archivo).convert("RGB"))
            self.mostrar(self.imgA, self.canvas_A)

    def cargar_B(self):
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png;*.bmp")])
        if archivo:
            imgB = Image.open(archivo).convert("RGB")
            if self.imgA is not None:
                # redimensionar al tamaño de A
                imgB = imgB.resize((self.imgA.shape[1], self.imgA.shape[0]))
            self.imgB = np.array(imgB)
            self.mostrar(self.imgB, self.canvas_B)

    def mostrar(self, arr, canvas):
        print("Mostrando imagen:", arr.shape, arr.dtype)  # debug
        im = Image.fromarray(arr.astype(np.uint8))
        im.thumbnail((250, 250))
        imtk = ImageTk.PhotoImage(im)
        canvas.config(image=imtk)
        canvas.image = imtk

        # ---- Operaciones RGB ----
    def suma_rgb_clamp(self):
        if self.imgA is None or self.imgB is None: return
        C = np.clip(self.imgA.astype(np.int16) + self.imgB.astype(np.int16), 0, 255).astype(np.uint8)
        self.mostrar(C, self.canvas_res)

    def resta_rgb_clamp(self):
        if self.imgA is None or self.imgB is None: return
        C = np.clip(self.imgA.astype(np.int16) - self.imgB.astype(np.int16), 0, 255).astype(np.uint8)
        self.mostrar(C, self.canvas_res)

    def suma_rgb_prom(self):
        if self.imgA is None or self.imgB is None: return
        suma = self.imgA.astype(np.int16) + self.imgB.astype(np.int16)
        resta = np.abs(self.imgA.astype(np.int16) - self.imgB.astype(np.int16))
        prom = ((suma + resta) / 2).clip(0,255).astype(np.uint8)
        self.mostrar(prom, self.canvas_res)

    def resta_rgb_prom(self):
        if self.imgA is None or self.imgB is None: return
        suma = self.imgA.astype(np.int16) + self.imgB.astype(np.int16)
        resta = np.abs(self.imgA.astype(np.int16) - self.imgB.astype(np.int16))
        prom = ((suma - resta) / 2).clip(0,255).astype(np.uint8)
        self.mostrar(prom, self.canvas_res)

    # ---- Operaciones YIQ ----
    def suma_yiq(self):
        if self.imgA is None or self.imgB is None: return
        yiqA = rgb_to_yiq(self.imgA)
        yiqB = rgb_to_yiq(self.imgB)
        yiq_sum = clamp_yiq(yiqA + yiqB)
        C = yiq_to_rgb(yiq_sum)
        self.mostrar(C, self.canvas_res)

    def resta_yiq(self):
        if self.imgA is None or self.imgB is None: return
        yiqA = rgb_to_yiq(self.imgA)
        yiqB = rgb_to_yiq(self.imgB)
        yiq_res = clamp_yiq(yiqA - yiqB)
        C = yiq_to_rgb(yiq_res)
        self.mostrar(C, self.canvas_res)

    # ---- Producto y cociente ----
    def producto(self):
        if self.imgA is None or self.imgB is None: return
        C = ((self.imgA.astype(np.float32) * self.imgB.astype(np.float32)) / 255).astype(np.uint8)
        self.mostrar(C, self.canvas_res)

    def cociente(self):
        if self.imgA is None or self.imgB is None: return
        C = (self.imgA.astype(np.float32) / (self.imgB.astype(np.float32)+1e-5) * 255).clip(0,255).astype(np.uint8)
        self.mostrar(C, self.canvas_res)

    # ---- Otros ----
    def resta_abs(self):
        if self.imgA is None or self.imgB is None: return
        C = np.abs(self.imgA.astype(np.int16) - self.imgB.astype(np.int16)).astype(np.uint8)
        self.mostrar(C, self.canvas_res)

    def if_lighter(self):
        if self.imgA is None or self.imgB is None: return
        YA = 0.299*self.imgA[:,:,0] + 0.587*self.imgA[:,:,1] + 0.114*self.imgA[:,:,2]
        YB = 0.299*self.imgB[:,:,0] + 0.587*self.imgB[:,:,1] + 0.114*self.imgB[:,:,2]
        mask = YA > YB
        C = np.where(mask[:,:,None], self.imgA, self.imgB)
        self.mostrar(C, self.canvas_res)

    def if_darker(self):
        if self.imgA is None or self.imgB is None: return
        YA = 0.299*self.imgA[:,:,0] + 0.587*self.imgA[:,:,1] + 0.114*self.imgA[:,:,2]
        YB = 0.299*self.imgB[:,:,0] + 0.587*self.imgB[:,:,1] + 0.114*self.imgB[:,:,2]
        mask = YA < YB
        C = np.where(mask[:,:,None], self.imgA, self.imgB)
        self.mostrar(C, self.canvas_res)


# ---- Main ----
if __name__ == "__main__":
    root = tk.Tk()
    app = PixelApp(root)
    root.mainloop()
