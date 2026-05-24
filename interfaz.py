# =========================================
# INTERFAZ - DETECTOR IA VS REAL
# =========================================

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from PIL import Image, ImageTk

import tensorflow as tf
import numpy as np

import actualizar_modelo
import threading

# =========================================
# CARGAR MODELO
# =========================================

modelo = tf.keras.models.load_model(
    "modelo/detector_ia.keras"
)

ruta_actual = None

# =========================================
# ANALIZAR IMAGEN
# =========================================

def analizar_imagen():

    global ruta_actual

    ruta_imagen = filedialog.askopenfilename(

        filetypes=[
            ("Imagenes",
            "*.jpg *.png *.jpeg")
        ]

    )

    if not ruta_imagen:

        return

    ruta_actual = ruta_imagen

    imagen = Image.open(
        ruta_imagen
    )

    imagen = imagen.convert(
        "RGB"
    )

    imagen_ia = imagen.resize(
        (128,128)
    )

    imagen_mostrar = imagen.resize(
        (250,250)
    )

    imagen_tk = ImageTk.PhotoImage(
        imagen_mostrar
    )

    label_imagen.config(
        image=imagen_tk
    )

    label_imagen.image = imagen_tk

    imagen_array = np.array(
        imagen_ia
    )

    imagen_array = (
        imagen_array / 255.0
    )

    imagen_array = np.expand_dims(

        imagen_array,

        axis=0

    )

    prediccion = modelo.predict(
        imagen_array,
        verbose=0
    )

    confianza = prediccion[0][0]

    if confianza > 0.5:

        resultado = "IMAGEN REAL"

        porcentaje = confianza*100

    else:

        resultado = (
            "IMAGEN GENERADA POR IA"
        )

        porcentaje = (
            1-confianza
        )*100

    texto_resultado.config(

        text=

        f"{resultado}\n"

        f"Confianza: "

        f"{porcentaje:.2f}%"

    )

# =========================================
# CORREGIR RESULTADO
# =========================================

def corregir():

    global ruta_actual

    if ruta_actual is None:

        messagebox.showerror(

            "Error",

            "Primero analiza una imagen"

        )

        return


    ventana2 = tk.Toplevel()

    ventana2.title(
        "Corregir Resultado"
    )

    ventana2.geometry(
        "300x220"
    )


    opcion = tk.StringVar()

    opcion.set(
        "real"
    )


    etiqueta = tk.Label(

        ventana2,

        text=

        "¿La imagen realmente es?",

        font=(

            "Arial",
            13

        )

    )

    etiqueta.pack(
        pady=10
    )


    real = tk.Radiobutton(

        ventana2,

        text="REAL",

        variable=opcion,

        value="real"

    )

    real.pack()


    ia = tk.Radiobutton(

        ventana2,

        text="IA",

        variable=opcion,

        value="ia"

    )

    ia.pack()


    def aprender():

        texto_resultado.config(

            text=

            "Aprendiendo..."

        )

        actualizar_modelo.agregar_imagen(

            ruta_actual,

            opcion.get()

        )

        hilo = threading.Thread(

            target=reentrenar

        )

        hilo.start()

        ventana2.destroy()


    boton_guardar = tk.Button(

        ventana2,

        text=

        "Agregar y Aprender",

        command=aprender,

        font=(

            "Arial",
            12

        )

    )

    boton_guardar.pack(

        pady=20

    )

# =========================================
# REENTRENAR
# =========================================

def reentrenar():

    global modelo

    actualizar_modelo.reentrenar()

    modelo = tf.keras.models.load_model(

        "modelo/detector_ia.keras"

    )

    texto_resultado.config(

        text=

        "Modelo actualizado"

    )

# =========================================
# VENTANA
# =========================================

ventana = tk.Tk()

ventana.title(

    "Detector IA vs Real"

)

ventana.geometry(

    "550x600"

)

# =========================================
# TITULO
# =========================================

titulo = tk.Label(

    ventana,

    text=

    "Detector IA vs Real",

    font=(

        "Arial",

        20

    )

)

titulo.pack(
    pady=10
)

# =========================================
# BOTON ANALIZAR
# =========================================

boton = tk.Button(

    ventana,

    text=

    "Seleccionar Imagen",

    command=

    analizar_imagen,

    font=(

        "Arial",

        14

    )

)

boton.pack(
    pady=10
)

# =========================================
# IMAGEN
# =========================================

label_imagen = tk.Label(
    ventana
)

label_imagen.pack(
    pady=10
)

# =========================================
# RESULTADO
# =========================================

texto_resultado = tk.Label(

    ventana,

    text=

    "Resultado aparecerá aquí",

    font=(

        "Arial",

        15

    )

)

texto_resultado.pack(
    pady=20
)

# =========================================
# BOTON NUEVO
# =========================================

boton_corregir = tk.Button(

    ventana,

    text=

    "Corregir Resultado",

    command=

    corregir,

    font=(

        "Arial",

        13

    ),

    bg="#FFCC66"

)

boton_corregir.pack(
    pady=10
)

# =========================================
# EJECUTAR
# =========================================

ventana.mainloop()
