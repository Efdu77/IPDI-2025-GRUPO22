import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from mi_libreria import apertura, cierre, borde_exterior, borde_interior, gradiente, mediana, erosion, dilatacion

class MorphologyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento Morfológico - TP5")
        self.root.geometry("1100x600")
        self.root.configure(bg="#2b2b2b")

        self.img_original = None
        self.img_resultado = None

        frame_top = tk.Frame(root, bg="#2b2b2b")
        frame_top.pack(pady=10)

        self.btn_cargar = ttk.Button(frame_top, text="Cargar imagen", command=self.cargar_imagen)
        self.btn_cargar.grid(row=0, column=0, padx=5)

        self.btn_aplicar = ttk.Button(frame_top, text="Aplicar operación", command=self.aplicar, state="disabled")
        self.btn_aplicar.grid(row=0, column=1, padx=5)

        self.btn_copiar = ttk.Button(frame_top, text="Copiar resultado → original", command=self.copiar_resultado, state="disabled")
        self.btn_copiar.grid(row=0, column=2, padx=5)

        self.btn_guardar = ttk.Button(frame_top, text="Guardar", command=self.guardar, state="disabled")
        self.btn_guardar.grid(row=0, column=3, padx=5)

        frame_ops = tk.Frame(root, bg="#2b2b2b")
        frame_ops.pack(pady=10)
        tk.Label(frame_ops, text="Operación morfológica:", fg="white", bg="#2b2b2b").grid(row=0, column=0)
        self.combo_op = ttk.Combobox(frame_ops, state="readonly", width=30, values=[
            "Erosión", "Dilatación", "Apertura", "Cierre",
            "Borde exterior", "Borde interior", "Gradiente", "Mediana"
        ])
        self.combo_op.grid(row=0, column=1, padx=5)
        self.combo_op.current(0)

        frame_imgs = tk.Frame(root, bg="#2b2b2b")
        frame_imgs.pack(pady=10)
        self.lbl1 = tk.Label(frame_imgs, text="Original", fg="white", bg="#2b2b2b")
        self.lbl1.grid(row=0, column=0)
        self.lbl2 = tk.Label(frame_imgs, text="Procesada", fg="white", bg="#2b2b2b")
        self.lbl2.grid(row=0, column=1)
        self.canvas1 = tk.Label(frame_imgs, bg="#1e1e1e")
        self.canvas1.grid(row=1, column=0, padx=10)
        self.canvas2 = tk.Label(frame_imgs, bg="#1e1e1e")
        self.canvas2.grid(row=1, column=1, padx=10)

    # Operaciones

    def cargar_imagen(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.bmp")])
        if not path:
            return
        self.img_original = Image.open(path).convert('L')
        self.mostrar(self.img_original, self.canvas1)
        self.btn_aplicar["state"] = "normal"

    def aplicar(self):
        if self.img_original is None:
            return
        op = self.combo_op.get()
        img_np = np.array(self.img_original)

        if op == "Erosión": res = erosion(img_np)
        elif op == "Dilatación": res = dilatacion(img_np)
        elif op == "Apertura": res = apertura(img_np)
        elif op == "Cierre": res = cierre(img_np)
        elif op == "Borde exterior": res = borde_exterior(img_np)
        elif op == "Borde interior": res = borde_interior(img_np)
        elif op == "Gradiente": res = gradiente(img_np)
        elif op == "Mediana": res = mediana(img_np)

        self.img_resultado = Image.fromarray(res)
        self.mostrar(self.img_resultado, self.canvas2)
        self.btn_copiar["state"] = "normal"
        self.btn_guardar["state"] = "normal"

    def copiar_resultado(self):
        if self.img_resultado:
            self.img_original = self.img_resultado.copy()
            self.mostrar(self.img_original, self.canvas1)

    def guardar(self):
        if self.img_resultado is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".bmp",
                                            filetypes=[("BMP", "*.bmp"), ("PNG", "*.png")])
        if not path:
            return
        self.img_resultado.save(path)
        messagebox.showinfo("Guardado", "Imagen procesada guardada correctamente.")

    def mostrar(self, img, widget):
        img_show = img.copy()
        img_show.thumbnail((450, 450))
        tk_img = ImageTk.PhotoImage(img_show)
        widget.configure(image=tk_img)
        widget.image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = MorphologyApp(root)
    root.mainloop()
