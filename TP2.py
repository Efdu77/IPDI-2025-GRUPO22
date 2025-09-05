import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class YIQEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor YIQ en tiempo real")

        # Variables
        self.img_original = None
        self.img_array = None
        self.img_proc = None

        # Botón para cargar imagen
        self.btn_cargar = tk.Button(root, text="Cargar Imagen", command=self.cargar_imagen)
        self.btn_cargar.pack(pady=5)

        # Lienzos para mostrar imágenes
        frame = tk.Frame(root)
        frame.pack()

        self.label_orig = tk.Label(frame, text="Imagen Original")
        self.label_orig.grid(row=0, column=0)
        self.label_proc = tk.Label(frame, text="Imagen Procesada")
        self.label_proc.grid(row=0, column=1)

        self.canvas_orig = tk.Label(frame)
        self.canvas_orig.grid(row=1, column=0, padx=10, pady=10)
        self.canvas_proc = tk.Label(frame)
        self.canvas_proc.grid(row=1, column=1, padx=10, pady=10)

        # Sliders independientes
        self.slider_Y = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1,
                                 orient="horizontal", label="Factor Y", command=self.actualizar)
        self.slider_Y.set(1.0)
        self.slider_Y.pack(fill="x")

        self.slider_I = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1,
                                 orient="horizontal", label="Factor I", command=self.actualizar)
        self.slider_I.set(1.0)
        self.slider_I.pack(fill="x")

        self.slider_Q = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1,
                                 orient="horizontal", label="Factor Q", command=self.actualizar)
        self.slider_Q.set(1.0)
        self.slider_Q.pack(fill="x")

        # Botón para guardar
        self.btn_guardar = tk.Button(root, text="Guardar Imagen",
                                     command=self.guardar_imagen, state="disabled")
        self.btn_guardar.pack(pady=5)

    def cargar_imagen(self):
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.png;*.jpeg")])
        if archivo:
            self.img_original = Image.open(archivo).convert("RGB")
            self.img_original.thumbnail((400, 400))
            self.img_array = np.array(self.img_original, dtype=np.float32) / 255.0

            self.mostrar_imagen(self.img_original, self.canvas_orig)
            self.actualizar()

    def mostrar_imagen(self, img_pil, widget):
        img_tk = ImageTk.PhotoImage(img_pil)
        widget.config(image=img_tk)
        widget.image = img_tk

    def actualizar(self, event=None):
        if self.img_array is None:
            return

        a = self.slider_Y.get()
        bI = self.slider_I.get()
        bQ = self.slider_Q.get()

        # Matriz RGB -> YIQ
        rgb_to_yiq = np.array([
            [0.299, 0.587, 0.114],
            [0.596, -0.275, -0.321],
            [0.212, -0.523, 0.311]
        ])
        yiq = np.tensordot(self.img_array, rgb_to_yiq.T, axes=1)

        # Aplicar factores con chequeo de rangos
        yiq[..., 0] = np.clip(a * yiq[..., 0], 0, 1)                # Y
        yiq[..., 1] = np.clip(bI * yiq[..., 1], -0.5957, 0.5957)    # I
        yiq[..., 2] = np.clip(bQ * yiq[..., 2], -0.5226, 0.5226)    # Q

        # Matriz YIQ -> RGB
        yiq_to_rgb = np.array([
            [1.0, 0.956, 0.621],
            [1.0, -0.272, -0.647],
            [1.0, -1.106, 1.703]
        ])
        rgb_proc = np.tensordot(yiq, yiq_to_rgb.T, axes=1)
        rgb_proc = np.clip(rgb_proc, 0, 1)

        # Guardar procesada
        self.img_proc = Image.fromarray((rgb_proc * 255).astype(np.uint8))

        # Mostrar
        self.mostrar_imagen(self.img_proc, self.canvas_proc)

        # Activar guardar
        self.btn_guardar.config(state="normal")

    def guardar_imagen(self):
        if self.img_proc:
            archivo = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
            if archivo:
                self.img_proc.save(archivo)


# Ejecutar
if __name__ == "__main__":
    root = tk.Tk()
    app = YIQEditor(root)
    root.mainloop()
