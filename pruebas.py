# =========================================
# DETECTOR DE IMÁGENES IA VS REALES
# pruebas.py
# =========================================

import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =========================================
# CARGAR MODELO
# =========================================

modelo = tf.keras.models.load_model("modelo/detector_ia.keras")

# =========================================
# FUNCIÓN PARA ANALIZAR IMAGEN
# =========================================

def analizar_imagen(ruta_imagen):

    # Abrir imagen
    imagen = Image.open(ruta_imagen)

    # Redimensionar
    imagen = imagen.resize((128, 128))

    # Convertir a array
    imagen_array = np.array(imagen)

    # Normalizar
    imagen_array = imagen_array / 255.0

    # Agregar dimensión
    imagen_array = np.expand_dims(imagen_array, axis=0)

    # Predicción
    prediccion = modelo.predict(imagen_array)

    confianza = prediccion[0][0]

    # Resultado
    if confianza >= 0.5:
        resultado = "REAL"
        porcentaje = confianza * 100
    else:
        resultado = "IA"
        porcentaje = (1 - confianza) * 100

    print("===================================")
    print("Imagen:", ruta_imagen)
    print("Resultado:", resultado)
    print("Confianza:", round(porcentaje, 2), "%")
    print("===================================\n")

# =========================================
# CARPETA DE PRUEBAS
# =========================================

carpeta_pruebas = "pruebas"

# =========================================
# RECORRER IMÁGENES
# =========================================

for archivo in os.listdir(carpeta_pruebas):

    if archivo.endswith(".jpg") or archivo.endswith(".png") or archivo.endswith(".jpeg"):

        ruta = os.path.join(carpeta_pruebas, archivo)

        analizar_imagen(ruta)
