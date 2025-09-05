import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

class YIQEditor:
    def __init__(self, root):
        self.root = root

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
        #Titulos de Imagenes
        self.label_orig = tk.Label(frame, text="Imagen Original")
        self.canvas_orig = tk.Label(frame)
        self.label_proc = tk.Label(frame, text="Imagen Procesada")
        self.canvas_proc = tk.Label(frame)
        self.label_pixel = tk.Label(frame, text="Pixel procesado")
        self.pixel_label = tk.Label(frame)

        # Modificadores de YIQ
        self.slider_Y = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1,
                                 orient="horizontal", label="Factor aY", command=self.actualizar)
        self.slider_Y.set(1.0)

        self.slider_I = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1,
                                 orient="horizontal", label="Factor bI", command=self.actualizar)
        self.slider_I.set(1.0)

        self.slider_Q = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1,
                                 orient="horizontal", label="Factor bQ", command=self.actualizar)
        self.slider_Q.set(1.0)

        # Botón para guardar
        self.btn_guardar = tk.Button(root, text="Guardar Imagen",
                                     command=self.guardar_imagen, state="disabled")

    def cargar_imagen(self):
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.png;*.jpeg")])
        if archivo:
            self.img_original = Image.open(archivo).convert("RGB")
            self.img_original.thumbnail((400, 400))
            self.img_array = np.array(self.img_original, dtype=np.float32) / 255.0

            self.mostrar_imagen(self.img_original, self.canvas_orig)
            self.btn_guardar.pack(pady=5)
            # Mostrar sliders
            self.slider_Y.pack(fill="x")
            self.slider_I.pack(fill="x")
            self.slider_Q.pack(fill="x")
            # Mostrar Titulos
            self.label_orig.grid(row=0, column=0)
            self.canvas_orig.grid(row=1, column=0, padx=10, pady=10)
            self.label_proc.grid(row=0, column=1)
            self.canvas_proc.grid(row=1, column=1, padx=10, pady=10)
            self.label_pixel.grid(row=0, column=2)
            self.pixel_label.grid(row=1, column=2, padx=10, pady=10)

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

                # --- Convertir un pixel a bytes y graficarlo ---
        pixel_bytes = tuple((np.array(self.img_proc)[0, 0]))  # primer pixel (0,0)
        img_pixel = Image.new("RGB", (50, 50), pixel_bytes)
        img_pixel_tk = ImageTk.PhotoImage(img_pixel)
        self.pixel_label.config(image=img_pixel_tk)
        self.pixel_label.image = img_pixel_tk


        # Activar guardar
        self.btn_guardar.config(state="normal")

    def guardar_imagen(self):
        if self.img_proc:
            archivo = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
            if archivo:
                self.img_proc.save(archivo)


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # Canvas para el contenido
        self.canvas = tk.Canvas(self)
        v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)

        # Frame interno que contendrá los widgets
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Actualizar región de scroll cuando cambia el tamaño del frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        # Meter el frame dentro del canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Configurar scrollbars
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        # Expandir el canvas
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)


# Ejecutar
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Editor RGB con YIQ ")

    scrollable = ScrollableFrame(root)
    scrollable.pack(fill="both", expand=True)

    app = YIQEditor(scrollable.scrollable_frame)

    root.mainloop()

