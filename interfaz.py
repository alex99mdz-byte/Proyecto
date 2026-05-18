# =========================================
# INTERFAZ - DETECTOR IA VS REAL
# interfaz.py
# =========================================

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# =========================================
# CARGAR MODELO
# =========================================

modelo = tf.keras.models.load_model(
    "modelo/detector_ia.keras"
)

# =========================================
# FUNCIÓN PARA ANALIZAR IMAGEN
# =========================================

def analizar_imagen():

    ruta_imagen = filedialog.askopenfilename(
        filetypes=[("Imagenes", "*.jpg *.png *.jpeg")]
    )

    if not ruta_imagen:
        return

    # Abrir imagen
    imagen = Image.open(ruta_imagen)
    imagen = imagen.resize((128,128))

    # Mostrar imagen
    imagen_mostrar = imagen.resize((250,250))
    imagen_tk = ImageTk.PhotoImage(imagen_mostrar)

    label_imagen.config(image=imagen_tk)
    label_imagen.image = imagen_tk

    # Convertir imagen para IA
    imagen_array = np.array(imagen) / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)

    # Predicción
    prediccion = modelo.predict(imagen_array)

    confianza = prediccion[0][0]

    # Resultado
    if confianza > 0.5:
        resultado = "IMAGEN REAL"
        porcentaje = confianza * 100
    else:
        resultado = "IMAGEN GENERADA POR IA"
        porcentaje = (1 - confianza) * 100

    texto_resultado.config(
        text=f"{resultado}\nConfianza: {porcentaje:.2f}%"
    )

# =========================================
# CREAR VENTANA
# =========================================

ventana = tk.Tk()
ventana.title("Detector IA vs Real")
ventana.geometry("500x500")

# =========================================
# TÍTULO
# =========================================

titulo = tk.Label(
    ventana,
    text="Detector de Imágenes IA",
    font=("Arial", 20)
)

titulo.pack(pady=10)

# =========================================
# BOTÓN
# =========================================

boton = tk.Button(
    ventana,
    text="Seleccionar Imagen",
    command=analizar_imagen,
    font=("Arial", 14)
)

boton.pack(pady=10)

# IMAGEN

label_imagen = tk.Label(ventana)
label_imagen.pack(pady=10)


# RESULTADO

texto_resultado = tk.Label(
    ventana,
    text="Resultado aparecerá aquí",
    font=("Arial", 16)
)

texto_resultado.pack(pady=20)

# EJECUTAR

ventana.mainloop()