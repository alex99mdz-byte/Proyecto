import tensorflow as tf
import shutil
import os

def agregar_imagen(
        ruta_imagen,
        categoria):

    carpeta = f"dataset/{categoria}"

    nombre = os.path.basename(
        ruta_imagen
    )

    destino = os.path.join(
        carpeta,
        nombre
    )

    shutil.copy(
        ruta_imagen,
        destino
    )

    print(
        "Imagen agregada"
    )

def reentrenar():

    modelo = tf.keras.models.load_model(
        "modelo/detector_ia.keras"
    )

    entrenamiento = tf.keras.utils.image_dataset_from_directory(

        "dataset",

        validation_split=0.2,

        subset="training",

        seed=123,

        image_size=(128,128),

        batch_size=32

    )

    validacion = tf.keras.utils.image_dataset_from_directory(

        "dataset",

        validation_split=0.2,

        subset="validation",

        seed=123,

        image_size=(128,128),

        batch_size=32

    )

    normalizar = tf.keras.layers.Rescaling(
        1./255
    )

    entrenamiento = entrenamiento.map(
        lambda x,y:
        (normalizar(x),y)
    )

    validacion = validacion.map(
        lambda x,y:
        (normalizar(x),y)
    )

    print(
        "Actualizando modelo..."
    )

    modelo.fit(

        entrenamiento,

        validation_data=validacion,

        epochs=2

    )

    modelo.save(
        "modelo/detector_ia.keras"
    )

    print(
        "Modelo actualizado"
    )


if __name__ == "__main__":

    ruta = input(
        "Ruta imagen corregida: "
    )

    categoria = input(
        "real o ia: "
    )

    agregar_imagen(
        ruta,
        categoria
    )

    reentrenar()
