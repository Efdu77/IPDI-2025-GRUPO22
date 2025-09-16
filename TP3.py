import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from mi_libreria import raiz, cuadrada, lineal_trozos, rgb_to_yiq, yiq_to_rgb

class AppFiltros:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtros de luminancia (según PDF)")
        # imagen original
        self.img = None
        # imagen procesada
        self.result = None
        # Frame de imágenes
        frame_imgs = tk.Frame(root)
        frame_imgs.pack(pady=10)
        tk.Label(frame_imgs, text="Original").grid(row=0, column=0)
        tk.Label(frame_imgs, text="Procesada").grid(row=0, column=1)
        self.canvas_orig = tk.Label(frame_imgs)
        self.canvas_orig.grid(row=1, column=0, padx=10)
        self.canvas_proc = tk.Label(frame_imgs)
        self.canvas_proc.grid(row=1, column=1, padx=10)
        # Controles
        frame_ctrl = tk.Frame(root)
        frame_ctrl.pack(pady=10)
        # Botón cargar
        tk.Button(frame_ctrl, text="Cargar Imagen", command=self.cargar_imagen).grid(row=0, column=0, padx=5)
        #Filtros
        tk.Label(frame_ctrl, text="Filtro:").grid(row=0, column=1, sticky="e")
        self.filtro_var = tk.StringVar()
        self.combo_filtro = ttk.Combobox(frame_ctrl, textvariable=self.filtro_var, state="readonly", width=20)
        self.combo_filtro['values'] = ["Raíz cuadrada", "Cuadrado", "Lineal a trozos"]
        self.combo_filtro.grid(row=0, column=2, padx=5)
        self.combo_filtro.bind("<<ComboboxSelected>>", self.on_filtro_change)
        # Botón aplicar
        tk.Button(frame_ctrl, text="Aplicar", command=self.aplicar).grid(row=0, column=3, padx=5)
        # Botón histograma
        tk.Button(frame_ctrl, text="Histograma", command=self.mostrar_histograma).grid(row=0, column=4, padx=5)
        # Botón guardar
        self.btn_guardar = tk.Button(frame_ctrl, text="Guardar Imagen", command=self.guardar_imagen, state="disabled")
        self.btn_guardar.grid(row=0, column=5, padx=5)
        #Sliders para el filtro "Lineal a trozos"
        self.frame_sliders = tk.Frame(root)
        tk.Label(self.frame_sliders, text="Ymin").grid(row=0, column=0)
        self.slider_ymin = tk.Scale(self.frame_sliders, from_=0.0, to=1.0, resolution=0.01,
                                    orient="horizontal", length=200)
        self.slider_ymin.set(0.2)
        self.slider_ymin.grid(row=0, column=1)
        tk.Label(self.frame_sliders, text="Ymax").grid(row=1, column=0)
        self.slider_ymax = tk.Scale(self.frame_sliders, from_=0.0, to=1.0, resolution=0.01,
                                    orient="horizontal", length=200)
        self.slider_ymax.set(0.8)
        self.slider_ymax.grid(row=1, column=1)

    def cargar_imagen(self):
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.png;*.bmp;*.jpeg")])
        if archivo:
            self.img = np.array(Image.open(archivo).convert("RGB"))
            self.mostrar(self.img, self.canvas_orig)
            self.result = None
            self.canvas_proc.config(image='') 
            self.btn_guardar.config(state="disabled")  

    def mostrar(self, arr, canvas):
        im = Image.fromarray(arr.astype(np.uint8))
        im.thumbnail((300, 300))
        imtk = ImageTk.PhotoImage(im)
        canvas.config(image=imtk)
        canvas.image = imtk

    def on_filtro_change(self, event=None):
        filtro = self.filtro_var.get()
        if filtro == "Lineal a trozos":
            self.frame_sliders.pack(pady=5)
        else:
            self.frame_sliders.pack_forget()

    def aplicar(self):
        if self.img is None:
            return
        filtro = self.filtro_var.get()
        if filtro == "Raíz cuadrada":
            self.result = aplicar_filtro(self.img, raiz)
        elif filtro == "Cuadrado":
            self.result = aplicar_filtro(self.img, cuadrada)
        elif filtro == "Lineal a trozos":
            ymin = float(self.slider_ymin.get())
            ymax = float(self.slider_ymax.get())
            if ymin >= ymax:
                tk.messagebox.showerror("Error", "Ymin debe ser menor que Ymax")
                return
            self.result = aplicar_filtro(self.img, lineal_trozos, Ymin=ymin, Ymax=ymax)
        else:
            tk.messagebox.showwarning("Aviso", "Seleccione un filtro antes de aplicar.")
            return
        if self.result is not None:
            self.mostrar(self.result, self.canvas_proc)
            self.btn_guardar.config(state="normal")

    def mostrar_histograma(self):
        if self.result is None:
            return
        img_norm = self.result / 255.0
        yiq = rgb_to_yiq(img_norm)
        Y = yiq[:,:,0].ravel()
        hist, bins = np.histogram(Y, bins=100, range=(0,1))
        hist = hist / hist.sum() * 100 
        plt.figure("Histograma de luminancia")
        plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]), color='gray', align='edge')
        plt.xlim(0,1)
        plt.ylim(0,100)
        plt.xlabel("Luminancia (0–1)")
        plt.ylabel("Frecuencia (%)")
        plt.title("Histograma de luminancia")
        plt.show()

    def guardar_imagen(self):
        if self.result is None:
            return
        archivo = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("JPEG", "*.jpg")])
        if archivo:
            Image.fromarray(self.result).save(archivo)


def aplicar_filtro(img, filtro, **kwargs):
    img_norm = img / 255.0
    yiq = rgb_to_yiq(img_norm)
    Y = yiq[:,:,0]
    yiq[:,:,0] = filtro(Y, **kwargs)
    rgb_proc = yiq_to_rgb(yiq)
    return np.clip(rgb_proc * 255, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    root = tk.Tk()
    app = AppFiltros(root)
    root.mainloop()
