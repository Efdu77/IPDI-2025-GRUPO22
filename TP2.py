import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

# Importar tu librería
from mi_libreria import (
    rgb_to_yiq, yiq_to_rgb,
    suma_rgb_clamp, suma_rgb_prom, resta_rgb_clamp, resta_rgb_prom,
    suma_yiq_clamp, suma_yiq_prom, resta_yiq_clamp, resta_yiq_prom,
    producto, cociente, resta_abs, if_lighter, if_darker
)

class PixelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aritmética de Píxeles - según PDF")

        # Variables para imágenes
        self.imgA = None
        self.imgB = None
        self.result = None  # para guardar resultado

        # ---- Frame superior: imágenes ----
        frame_imgs = tk.Frame(root)
        frame_imgs.pack(pady=10)

        tk.Label(frame_imgs, text="Imagen A").grid(row=0, column=0)
        tk.Label(frame_imgs, text="Resultado").grid(row=0, column=1)
        tk.Label(frame_imgs, text="Imagen B").grid(row=0, column=2)

        self.canvas_A = tk.Label(frame_imgs)
        self.canvas_A.grid(row=1, column=0, padx=10)
        self.canvas_res = tk.Label(frame_imgs)
        self.canvas_res.grid(row=1, column=1, padx=10)
        self.canvas_B = tk.Label(frame_imgs)
        self.canvas_B.grid(row=1, column=2, padx=10)

        # Botón de guardar debajo del resultado
        self.btn_guardar = tk.Button(frame_imgs, text="Guardar resultado", command=self.guardar_resultado, state="disabled")
        self.btn_guardar.grid(row=2, column=1, pady=5)

        # ---- Frame para carga de imágenes (siempre visible) ----
        frame_load = tk.Frame(root)
        frame_load.pack(pady=10)

        tk.Button(frame_load, text="Cargar Imagen A", command=self.cargar_A).grid(row=0, column=0, padx=5)
        tk.Button(frame_load, text="Cargar Imagen B", command=self.cargar_B).grid(row=0, column=1, padx=5)

        # ---- Frame para controles (oculto al inicio) ----
        self.frame_ctrl = tk.Frame(root)

        # Desplegable de operaciones
        tk.Label(self.frame_ctrl, text="Operación:").grid(row=0, column=0, sticky="e")
        self.op_var = tk.StringVar()
        self.combo_op = ttk.Combobox(self.frame_ctrl, textvariable=self.op_var, state="readonly", width=20)
        self.combo_op['values'] = [
            "Suma", "Resta",
            "Producto", "Cociente",
            "Resta Absoluta",
            "If-lighter", "If-darker"
        ]
        self.combo_op.grid(row=0, column=1, padx=5)
        self.combo_op.bind("<<ComboboxSelected>>", self.on_op_change)

        # Desplegable de espacio/variante (solo válido para suma y resta)
        tk.Label(self.frame_ctrl, text="Modo:").grid(row=1, column=0, sticky="e")
        self.mode_var = tk.StringVar()
        self.combo_mode = ttk.Combobox(self.frame_ctrl, textvariable=self.mode_var, state="disabled", width=20)
        self.combo_mode['values'] = [
            "RGB Clampeada", "RGB Promediada",
            "YIQ Clampeada", "YIQ Promediada"
        ]
        self.combo_mode.grid(row=1, column=1, padx=5)

        # Botón aplicar
        self.btn_aplicar = tk.Button(self.frame_ctrl, text="Aplicar", command=self.aplicar_operacion)
        self.btn_aplicar.grid(row=2, column=0, columnspan=2, pady=10)

        # Ocultar frame_ctrl al inicio
        self.frame_ctrl.pack_forget()

    # ---- Funciones de carga ----
    def cargar_A(self):
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.png;*.bmp")])
        if archivo:
            self.imgA = np.array(Image.open(archivo).convert("RGB"))
            self.mostrar(self.imgA, self.canvas_A)
            self.mostrar_controles()

    def cargar_B(self):
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.png;*.bmp")])
        if archivo:
            imgB = Image.open(archivo).convert("RGB")
            if self.imgA is not None:
                imgB = imgB.resize((self.imgA.shape[1], self.imgA.shape[0]))
            self.imgB = np.array(imgB)
            self.mostrar(self.imgB, self.canvas_B)
            self.mostrar_controles()

    def mostrar_controles(self):
        if self.imgA is not None and self.imgB is not None:
            self.frame_ctrl.pack(pady=10)

    # ---- Mostrar imagen en Tkinter ----
    def mostrar(self, arr, canvas):
        im = Image.fromarray(arr.astype(np.uint8))
        im.thumbnail((250, 250))
        imtk = ImageTk.PhotoImage(im)
        canvas.config(image=imtk)
        canvas.image = imtk

    # ---- Cambiar estado del combobox de modo ----
    def on_op_change(self, event=None):
        op = self.op_var.get()
        if op in ["Suma", "Resta"]:
            self.combo_mode.config(state="readonly")
            if not self.mode_var.get():
                self.combo_mode.set("RGB Clampeada")
        else:
            self.combo_mode.set("")
            self.combo_mode.config(state="disabled")

    # ---- Aplicar operación seleccionada ----
    def aplicar_operacion(self):
        if self.imgA is None or self.imgB is None:
            return

        op = self.op_var.get()
        mode = self.mode_var.get()
        result = None

        if op == "Suma":
            if mode == "RGB Clampeada":
                result = suma_rgb_clamp(self.imgA, self.imgB)
            elif mode == "RGB Promediada":
                result = suma_rgb_prom(self.imgA, self.imgB)
            elif mode == "YIQ Clampeada":
                yiqA, yiqB = rgb_to_yiq(self.imgA), rgb_to_yiq(self.imgB)
                result = yiq_to_rgb(suma_yiq_clamp(yiqA, yiqB))
            elif mode == "YIQ Promediada":
                yiqA, yiqB = rgb_to_yiq(self.imgA), rgb_to_yiq(self.imgB)
                result = yiq_to_rgb(suma_yiq_prom(yiqA, yiqB))

        elif op == "Resta":
            if mode == "RGB Clampeada":
                result = resta_rgb_clamp(self.imgA, self.imgB)
            elif mode == "RGB Promediada":
                result = resta_rgb_prom(self.imgA, self.imgB)
            elif mode == "YIQ Clampeada":
                yiqA, yiqB = rgb_to_yiq(self.imgA), rgb_to_yiq(self.imgB)
                result = yiq_to_rgb(resta_yiq_clamp(yiqA, yiqB))
            elif mode == "YIQ Promediada":
                yiqA, yiqB = rgb_to_yiq(self.imgA), rgb_to_yiq(self.imgB)
                result = yiq_to_rgb(resta_yiq_prom(yiqA, yiqB))

        elif op == "Producto":
            result = producto(self.imgA, self.imgB)

        elif op == "Cociente":
            result = cociente(self.imgA, self.imgB)

        elif op == "Resta Absoluta":
            result = resta_abs(self.imgA, self.imgB)

        elif op == "If-lighter":
            result = if_lighter(self.imgA, self.imgB)

        elif op == "If-darker":
            result = if_darker(self.imgA, self.imgB)

        if result is not None:
            self.result = result
            self.mostrar(result, self.canvas_res)
            self.btn_guardar.config(state="normal")  # habilitar guardar

    # ---- Guardar resultado ----
    def guardar_resultado(self):
        if self.result is None:
            return
        archivo = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("JPEG", "*.jpg")])
        if archivo:
            Image.fromarray(self.result).save(archivo)

# ---- Main ----
if __name__ == "__main__":
    root = tk.Tk()
    app = PixelApp(root)
    root.mainloop()
