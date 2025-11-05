import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from mi_libreria import binarizacion_50, binarizacion_dos_modas, binarizacion_otsu, borde_laplaciano, erosion, dilatacion, borde_morfologico, borde_marching_squares, color_fill

class TP6:
    def __init__(self, root):
        self.root = root
        self.root.title("TP5 - Segmentación y Binarización")
        self.root.geometry("1100x700")
        self.root.configure(bg="#2b2b2b")

        self.img_original = None
        self.img_resultado = None
        self.seed = None

        # === Controles superiores ===
        frame_top = tk.Frame(root, bg="#2b2b2b")
        frame_top.pack(pady=10)

        self.btn_cargar = ttk.Button(frame_top, text="Cargar imagen", command=self.cargar)
        self.btn_cargar.grid(row=0, column=0, padx=5)

        self.combo_op = ttk.Combobox(frame_top, state="readonly", width=35, values=[
            "Binarización 50%",
            "Binarización Dos Modas",
            "Binarización Otsu",
            "Borde Laplaciano",
            "Borde Morfológico",
            "Marching Squares",
            "Color Fill (Varita Mágica)"
        ])
        self.combo_op.current(0)
        self.combo_op.grid(row=0, column=1, padx=5)

        self.btn_aplicar = ttk.Button(frame_top, text="Aplicar", command=self.aplicar, state="disabled")
        self.btn_aplicar.grid(row=0, column=2, padx=5)

        self.btn_guardar = ttk.Button(frame_top, text="Guardar", command=self.guardar, state="disabled")
        self.btn_guardar.grid(row=0, column=3, padx=5)

        # === Área imágenes ===
        frame_imgs = tk.Frame(root, bg="#2b2b2b")
        frame_imgs.pack(pady=20)

        self.lbl1 = tk.Label(frame_imgs, text="Original", fg="white", bg="#2b2b2b")
        self.lbl1.grid(row=0, column=0)
        self.lbl2 = tk.Label(frame_imgs, text="Resultado", fg="white", bg="#2b2b2b")
        self.lbl2.grid(row=0, column=1)

        self.canvas1 = tk.Label(frame_imgs, bg="#1e1e1e")
        self.canvas1.grid(row=1, column=0, padx=10)
        self.canvas2 = tk.Label(frame_imgs, bg="#1e1e1e")
        self.canvas2.grid(row=1, column=1, padx=10)

        self.canvas1.bind("<Button-1>", self.click_varita)

    def cargar(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.bmp")])
        if not path:
            return
        img = Image.open(path).convert("L")
        self.img_original = np.array(img)
        self.mostrar(img, self.canvas1)
        self.btn_aplicar["state"] = "normal"

    def aplicar(self):
        if self.img_original is None:
            return
        op = self.combo_op.get()
        img = self.img_original.copy()

        if op == "Binarización 50%":
            res = operaciones.binarizacion_50(img)
        elif op == "Binarización Dos Modas":
            res = operaciones.binarizacion_dos_modas(img)
        elif op == "Binarización Otsu":
            res = operaciones.binarizacion_otsu(img)
        elif op == "Borde Laplaciano":
            res = operaciones.borde_laplaciano(img)
        elif op == "Borde Morfológico":
            res = operaciones.borde_morfologico(img)
        elif op == "Marching Squares":
            res = operaciones.borde_marching_squares(img)
        elif op == "Color Fill (Varita Mágica)":
            if self.seed is None:
                messagebox.showinfo("Varita mágica", "Haz clic en la imagen para seleccionar el punto semilla.")
                return
            x, y = self.seed
            res = operaciones.color_fill(img, x, y, tolerancia=20)
            self.seed = None
        else:
            messagebox.showerror("Error", "Operación no reconocida.")
            return

        self.img_resultado = res
        self.mostrar(Image.fromarray(res), self.canvas2)
        self.btn_guardar["state"] = "normal"

    def click_varita(self, event):
        if self.combo_op.get() != "Color Fill (Varita Mágica)":
            return
        if self.img_original is not None:
            img_h, img_w = self.img_original.shape
            x_real = int(event.x * img_w / self.canvas1.winfo_width())
            y_real = int(event.y * img_h / self.canvas1.winfo_height())
            self.seed = (x_real, y_real)
            self.aplicar()

    def guardar(self):
        if self.img_resultado is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".bmp",
                                            filetypes=[("BMP", "*.bmp"), ("PNG", "*.png")])
        if not path:
            return
        Image.fromarray(self.img_resultado).save(path)
        messagebox.showinfo("Guardado", "Imagen guardada correctamente.")

    def mostrar(self, img, widget):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.thumbnail((450, 450))
        tk_img = ImageTk.PhotoImage(img)
        widget.configure(image=tk_img)
        widget.image = tk_img


if __name__ == "__main__":
    root = tk.Tk()
    app = TP5App(root)
    root.mainloop()
