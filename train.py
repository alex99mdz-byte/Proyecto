# DETECTOR DE IMÁGENES IA VS REALES
# train.py

import tensorflow as tf
import matplotlib.pyplot as plt

# CARGAR DATASET

dataset_entrenamiento = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

dataset_validacion = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)
# NOMBRES DE CLASES

nombres_clases = dataset_entrenamiento.class_names

print("Clases detectadas:")
print(nombres_clases)

# NORMALIZAR IMÁGENES

normalizacion = tf.keras.layers.Rescaling(1./255)

dataset_entrenamiento = dataset_entrenamiento.map(
    lambda x, y: (normalizacion(x), y)
)

dataset_validacion = dataset_validacion.map(
    lambda x, y: (normalizacion(x), y)
)

# CREAR MODELO IA

capa1 = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3,3),
    activation='relu',
    input_shape=(128,128,3)
)

pool1 = tf.keras.layers.MaxPooling2D(2,2)

capa2 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(3,3),
    activation='relu'
)

pool2 = tf.keras.layers.MaxPooling2D(2,2)

aplanar = tf.keras.layers.Flatten()

oculta = tf.keras.layers.Dense(
    units=128,
    activation='relu'
)

salida = tf.keras.layers.Dense(
    units=1,
    activation='sigmoid'
)

modelo = tf.keras.Sequential([
    capa1,
    pool1,
    capa2,
    pool2,
    aplanar,
    oculta,
    salida
])

# COMPILAR MODELO

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ENTRENAR MODELO

print("Iniciando entrenamiento...")

historial = modelo.fit(
    dataset_entrenamiento,
    validation_data=dataset_validacion,
    epochs=10
)

print("Modelo entrenado")

# GUARDAR MODELO

modelo.save("modelo/detector_ia.keras")

print("Modelo guardado correctamente")

# GRÁFICA DE PRECISIÓN

plt.xlabel('Épocas')
plt.ylabel('Precisión')

plt.plot(historial.history['accuracy'])
plt.plot(historial.history['val_accuracy'])

plt.legend(['Entrenamiento', 'Validación'])

plt.title("Precisión del modelo")

plt.show()

# MOSTRAR PESOS DEL MODELO

print("Pesos internos del modelo")

print(capa1.get_weights())
print(capa2.get_weights())
print(oculta.get_weights())
print(salida.get_weights())