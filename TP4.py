import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from mi_libreria import pasabajo_plano_3x3, pasabajo_plano_5x5, pasabajo_plano_7x7, pasabajo_bartlett_3x3, pasabajo_bartlett_5x5, pasabajo_bartlett_7x7, pasabajo_gaussiano_5x5, pasabajo_gaussiano_7x7, laplaciano_v4, laplaciano_v8, sobel_este, sobel_ne, sobel_no, sobel_norte, sobel_oeste, sobel_se, sobel_so, sobel_sur, pasabanda_02, pasabanda_04, pasaaltos_02, pasaaltos_04

class TP4App:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento por Convolución - TP4")
        self.root.geometry("1100x600")
        self.root.configure(bg="#2b2b2b")

        self.img_original = None
        self.img_resultado = None

        frame_top = tk.Frame(root, bg="#2b2b2b")
        frame_top.pack(pady=10)

        self.btn_cargar = ttk.Button(frame_top, text="Cargar imagen", command=self.cargar_imagen)
        self.btn_cargar.grid(row=0, column=0, padx=5)

        self.btn_aplicar = ttk.Button(frame_top, text="Aplicar filtro", command=self.aplicar_filtro, state="disabled")
        self.btn_aplicar.grid(row=0, column=1, padx=5)

        self.btn_guardar = ttk.Button(frame_top, text="Guardar resultado", command=self.guardar_resultado, state="disabled")
        self.btn_guardar.grid(row=0, column=2, padx=5)

        frame_select = tk.Frame(root, bg="#2b2b2b")
        frame_select.pack(pady=10)

        tk.Label(frame_select, text="Tipo de filtro:", fg="white", bg="#2b2b2b").grid(row=0, column=0, padx=5)
        self.combo_tipo = ttk.Combobox(frame_select, state="readonly", width=30)
        self.combo_tipo.grid(row=0, column=1, padx=5)

        tk.Label(frame_select, text="Variante:", fg="white", bg="#2b2b2b").grid(row=0, column=2, padx=5)
        self.combo_variante = ttk.Combobox(frame_select, state="readonly", width=30)
        self.combo_variante.grid(row=0, column=3, padx=5)

        self.filtros = {
            "Pasabajos": [
                "Plano 3x3", "Plano 5x5", "Plano 7x7",
                "Bartlett 3x3", "Bartlett 5x5", "Bartlett 7x7",
                "Gaussiano 5x5", "Gaussiano 7x7"
            ],
            "Detectores de bordes": [
                "Laplaciano v4", "Laplaciano v8",
                "Sobel Norte", "Sobel Sur", "Sobel Este", "Sobel Oeste",
                "Sobel NE", "Sobel NO", "Sobel SE", "Sobel SO"
            ],
            "Pasabanda": [
                "Frecuencia 0.2", "Frecuencia 0.4"
            ],
            "Pasaaltos": [
                "Frecuencia 0.2", "Frecuencia 0.4"
            ]
        }

        self.combo_tipo["values"] = list(self.filtros.keys())
        self.combo_tipo.bind("<<ComboboxSelected>>", self.actualizar_variantes)

        frame_imgs = tk.Frame(root, bg="#2b2b2b")
        frame_imgs.pack(pady=10)

        self.lbl_original = tk.Label(frame_imgs, text="Original", fg="white", bg="#2b2b2b")
        self.lbl_original.grid(row=0, column=0)

        self.lbl_resultado = tk.Label(frame_imgs, text="Filtrada", fg="white", bg="#2b2b2b")
        self.lbl_resultado.grid(row=0, column=1)

        self.canvas_original = tk.Label(frame_imgs, bg="#1e1e1e")
        self.canvas_original.grid(row=1, column=0, padx=10)

        self.canvas_resultado = tk.Label(frame_imgs, bg="#1e1e1e")
        self.canvas_resultado.grid(row=1, column=1, padx=10)

    #Funciones

    def cargar_imagen(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.bmp")])
        if not path:
            return
        self.img_original = Image.open(path).convert('L')
        self.mostrar_imagen(self.img_original, self.canvas_original)
        self.btn_aplicar["state"] = "normal"
        self.combo_tipo["state"] = "readonly"
        self.combo_variante["state"] = "readonly"

    def actualizar_variantes(self, event=None):
        tipo = self.combo_tipo.get()
        self.combo_variante["values"] = self.filtros[tipo]
        self.combo_variante.current(0)

    def aplicar_filtro(self):
        if self.img_original is None:
            return

        tipo = self.combo_tipo.get()
        variante = self.combo_variante.get()
        img_np = np.array(self.img_original)

        try:
            #Pasabajos
            if tipo == "Pasabajos":
                if variante == "Plano 3x3": res = pasabajo_plano_3x3(img_np)
                elif variante == "Plano 5x5": res = pasabajo_plano_5x5(img_np)
                elif variante == "Plano 7x7": res = pasabajo_plano_7x7(img_np)
                elif variante == "Bartlett 3x3": res = pasabajo_bartlett_3x3(img_np)
                elif variante == "Bartlett 5x5": res = pasabajo_bartlett_5x5(img_np)
                elif variante == "Bartlett 7x7": res = pasabajo_bartlett_7x7(img_np)
                elif variante == "Gaussiano 5x5": res = pasabajo_gaussiano_5x5(img_np)
                elif variante == "Gaussiano 7x7": res = pasabajo_gaussiano_7x7(img_np)

            #Bordes
            elif tipo == "Detectores de bordes":
                if variante == "Laplaciano v4": res = laplaciano_v4(img_np)
                elif variante == "Laplaciano v8": res = laplaciano_v8(img_np)
                #Bordes 8 direcciones
                elif variante == "Sobel Norte": res = sobel_norte(img_np)
                elif variante == "Sobel Sur": res = sobel_sur(img_np)
                elif variante == "Sobel Este": res = sobel_este(img_np)
                elif variante == "Sobel Oeste": res = sobel_oeste(img_np)
                elif variante == "Sobel NE": res = sobel_ne(img_np)
                elif variante == "Sobel NO": res = sobel_no(img_np)
                elif variante == "Sobel SE": res = sobel_se(img_np)
                elif variante == "Sobel SO": res = sobel_so(img_np)

            #Pasabanda
            elif tipo == "Pasabanda":
                if variante == "Frecuencia 0.2": res = pasabanda_02(img_np)
                elif variante == "Frecuencia 0.4": res = pasabanda_04(img_np)

            #Pasaaltos
            elif tipo == "Pasaaltos":
                if variante == "Frecuencia 0.2": res = pasaaltos_02(img_np)
                elif variante == "Frecuencia 0.4": res = pasaaltos_04(img_np)

            self.img_resultado = Image.fromarray(res)
            self.mostrar_imagen(self.img_resultado, self.canvas_resultado)
            self.btn_guardar["state"] = "normal"

        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar filtro:\n{e}")

    def mostrar_imagen(self, img, widget):
        img_resized = img.copy()
        img_resized.thumbnail((450, 450))
        tk_img = ImageTk.PhotoImage(img_resized)
        widget.configure(image=tk_img)
        widget.image = tk_img

    def guardar_resultado(self):
        if self.img_resultado is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("BMP", "*.bmp"), ("PNG", "*.png")])
        if not path:
            return
        self.img_resultado.save(path)
        messagebox.showinfo("Guardado", "Imagen filtrada guardada correctamente.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TP4App(root)
    root.mainloop()
