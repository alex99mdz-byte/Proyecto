# =========================================
# DETECTOR DE IMÁGENES IA VS REALES
# prediccion.py
# =========================================

import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =========================================
# CARGAR MODELO
# =========================================

modelo = tf.keras.models.load_model(
    "modelo/detector_ia.keras"
)

# =========================================
# FUNCIÓN DE PREDICCIÓN
# =========================================

def analizar_imagen(ruta_imagen):

    if not os.path.exists(ruta_imagen):

        return (
            "ERROR",
            0
        )

    imagen = Image.open(
        ruta_imagen
    )

    imagen = imagen.convert(
        "RGB"
    )

    imagen = imagen.resize(
        (128,128)
    )

    imagen_array = np.array(
        imagen
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

    if confianza >= 0.5:

        resultado = "REAL"

        porcentaje = confianza*100

    else:

        resultado = "IA"

        porcentaje = (
            1-confianza
        )*100

    return (
        resultado,
        round(
            porcentaje,
            2
        )
    )

# =========================================
# PRUEBA MANUAL
# =========================================

if __name__ == "__main__":

    ruta = input(
        "Ruta imagen: "
    )

    resultado, confianza = analizar_imagen(
        ruta
    )

    print()

    print(
        "Resultado:",
        resultado
    )

    print(
        "Confianza:",
        confianza,
        "%"
    )
