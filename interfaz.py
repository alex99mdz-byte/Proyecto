# =========================================
# INTERFAZ PROFESIONAL - DETECTOR IA VS REAL
# ESTILO ACADÉMICO - ESCALA DE GRISES Y AZULES
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
# ESTILOS PROFESIONALES ACADÉMICOS
# =========================================

class AcademicStyles:
    COLORS = {
        'bg_primary': '#FFFFFF',        # Blanco principal
        'bg_secondary': '#F5F5F5',      # Gris muy claro
        'bg_tertiary': '#EBEBEB',       # Gris claro
        'bg_dark': '#E0E0E0',           # Gris medio claro
        
        'primary_blue': '#1E3A8A',      # Azul profundo
        'primary_blue_dark': '#172554',  # Azul más oscuro
        'primary_blue_light': '#2563EB', # Azul brillante
        'accent_blue': '#3B82F6',        # Azul acento
        
        'text_primary': '#1A1A1A',      # Casi negro
        'text_secondary': '#4A4A4A',    # Gris oscuro
        'text_tertiary': '#6B6B6B',     # Gris medio
        'text_light': '#9A9A9A',         # Gris claro
        
        'success': '#059669',            # Verde profesional
        'error': '#DC2626',              # Rojo profesional
        'warning': '#D97706',            # Naranja profesional
        'border': '#D1D5DB',             # Gris borde
        'shadow': '#E5E7EB',             # Gris sombra
        'hover': '#1D4ED8'               # Azul hover
    }
    
    FONTS = {
        'title': ('Segoe UI', 20, 'bold'),
        'subtitle': ('Segoe UI', 12, 'bold'),
        'body': ('Segoe UI', 10),
        'result': ('Segoe UI', 12, 'bold'),
        'button': ('Segoe UI', 10, 'bold'),
        'small': ('Segoe UI', 9)
    }

class ProfessionalButton(tk.Button):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(
            relief=tk.FLAT,
            cursor="hand2",
            borderwidth=0,
            highlightthickness=0,
            padx=15,
            pady=8
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_enter(self, e):
        self.configure(bg=AcademicStyles.COLORS['hover'])
        
    def on_leave(self, e):
        self.configure(bg=AcademicStyles.COLORS['primary_blue'])

class SecondaryButton(tk.Button):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(
            relief=tk.FLAT,
            cursor="hand2",
            borderwidth=1,
            highlightthickness=0,
            padx=15,
            pady=8
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_enter(self, e):
        self.configure(bg=AcademicStyles.COLORS['bg_tertiary'])
        
    def on_leave(self, e):
        self.configure(bg=AcademicStyles.COLORS['bg_secondary'])

class CardFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(
            bg=AcademicStyles.COLORS['bg_secondary'],
            relief=tk.RAISED,
            bd=1,
            highlightthickness=1,
            highlightbackground=AcademicStyles.COLORS['border'],
            highlightcolor=AcademicStyles.COLORS['border']
        )

# =========================================
# ANALIZAR IMAGEN
# =========================================

def analizar_imagen():
    global ruta_actual
    
    # Estado de carga
    texto_resultado.config(
        text="Analizando imagen...",
        fg=AcademicStyles.COLORS['primary_blue_light']
    )
    ventana.update()
    
    ruta_imagen = filedialog.askopenfilename(
        filetypes=[
            ("Imagenes", "*.jpg *.png *.jpeg")
        ]
    )
    
    if not ruta_imagen:
        texto_resultado.config(
            text="Esperando imagen...",
            fg=AcademicStyles.COLORS['text_secondary']
        )
        return
    
    ruta_actual = ruta_imagen
    
    # Abrir y redimensionar imagen manteniendo proporción
    imagen_original = Image.open(ruta_imagen)
    imagen_original = imagen_original.convert("RGB")
    
    # Tamaño deseado para la imagen
    TAMANO_IMAGEN = (450, 450)
    
    # Redimensionar manteniendo la proporción
    imagen_redimensionada = imagen_original.copy()
    imagen_redimensionada.thumbnail(TAMANO_IMAGEN, Image.Resampling.LANCZOS)
    
    # Crear un marco cuadrado del tamaño exacto
    marco_size = TAMANO_IMAGEN[0] + 20
    imagen_con_marco = Image.new('RGB', (marco_size, marco_size), AcademicStyles.COLORS['border'])
    
    # Calcular posición para centrar la imagen
    x_offset = (marco_size - imagen_redimensionada.width) // 2
    y_offset = (marco_size - imagen_redimensionada.height) // 2
    
    # Pegar la imagen redimensionada en el centro
    imagen_con_marco.paste(imagen_redimensionada, (x_offset, y_offset))
    
    # Convertir a PhotoImage
    imagen_tk = ImageTk.PhotoImage(imagen_con_marco)
    
    # Actualizar el label con el tamaño fijo
    label_imagen.config(image=imagen_tk, width=marco_size, height=marco_size)
    label_imagen.image = imagen_tk
    
    # Procesar para IA
    imagen_ia = imagen_original.resize((128, 128), Image.Resampling.LANCZOS)
    imagen_array = np.array(imagen_ia)
    imagen_array = (imagen_array / 255.0)
    imagen_array = np.expand_dims(imagen_array, axis=0)
    
    # Animación de análisis
    for i in range(3):
        texto_resultado.config(text=f"Analizando{'.' * (i+1)}")
        ventana.update()
        ventana.after(100)
    
    prediccion = modelo.predict(imagen_array, verbose=0)
    confianza = prediccion[0][0]
    
    # Resultado
    if confianza > 0.5:
        resultado = "IMAGEN REAL"
        porcentaje = confianza * 100
        color = AcademicStyles.COLORS['success']
    else:
        resultado = "IMAGEN GENERADA POR IA"
        porcentaje = (1 - confianza) * 100
        color = AcademicStyles.COLORS['error']
    
    texto_resultado.config(
        text=f"{resultado}\nConfianza: {porcentaje:.2f}%",
        fg=color
    )
    
    # Barra de confianza
    num_cuadros = int(porcentaje / 10)
    barra_texto = "█" * num_cuadros + "▒" * (10 - num_cuadros)
    barra_confianza.config(text=barra_texto, fg=color)
    
    # Actualizar el marco
    marco_imagen.config(bg=AcademicStyles.COLORS['primary_blue'])

# =========================================
# CORREGIR RESULTADO
# =========================================

def corregir():
    global ruta_actual
    
    if ruta_actual is None:
        messagebox.showerror(
            "Error",
            "Primero analice una imagen antes de corregir."
        )
        return
    
    # Ventana secundaria
    ventana2 = tk.Toplevel()
    ventana2.title("Corrección de Resultado")
    ventana2.geometry("450x400")
    ventana2.configure(bg=AcademicStyles.COLORS['bg_primary'])
    
    # Marco principal
    marco = CardFrame(ventana2)
    marco.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    etiqueta = tk.Label(
        marco,
        text="CORRECCIÓN DE RESULTADO",
        font=AcademicStyles.FONTS['subtitle'],
        fg=AcademicStyles.COLORS['text_primary'],
        bg=AcademicStyles.COLORS['bg_secondary']
    )
    etiqueta.pack(pady=20)
    
    # Información de la imagen
    if ruta_actual:
        nombre_archivo = ruta_actual.split('/')[-1]
        info_label = tk.Label(
            marco,
            text=f"Imagen: {nombre_archivo}",
            font=AcademicStyles.FONTS['small'],
            fg=AcademicStyles.COLORS['text_tertiary'],
            bg=AcademicStyles.COLORS['bg_secondary']
        )
        info_label.pack(pady=5)
    
    opcion = tk.StringVar()
    opcion.set("real")
    
    # Radio buttons profesionales
    style = {
        "bg": AcademicStyles.COLORS['bg_secondary'],
        "fg": AcademicStyles.COLORS['text_primary'],
        "selectcolor": AcademicStyles.COLORS['primary_blue_light'],
        "font": AcademicStyles.FONTS['body'],
        "activebackground": AcademicStyles.COLORS['bg_secondary'],
        "activeforeground": AcademicStyles.COLORS['primary_blue']
    }
    
    real = tk.Radiobutton(
        marco,
        text="Imagen Real",
        variable=opcion,
        value="real",
        **style
    )
    real.pack(pady=10)
    
    ia = tk.Radiobutton(
        marco,
        text="Imagen Generada por IA",
        variable=opcion,
        value="ia",
        **style
    )
    ia.pack(pady=10)
    
    def aprender():
        texto_resultado.config(
            text="Aprendiendo...",
            fg=AcademicStyles.COLORS['primary_blue_light']
        )
        
        actualizar_modelo.agregar_imagen(
            ruta_actual,
            opcion.get()
        )
        
        hilo = threading.Thread(target=reentrenar)
        hilo.start()
        ventana2.destroy()
    
    boton_guardar = ProfessionalButton(
        marco,
        text="Agregar y Reentrenar",
        command=aprender,
        font=AcademicStyles.FONTS['button'],
        bg=AcademicStyles.COLORS['primary_blue'],
        fg=AcademicStyles.COLORS['bg_primary']
    )
    boton_guardar.pack(pady=20)

# =========================================
# REENTRENAR
# =========================================

def reentrenar():
    global modelo
    
    actualizar_modelo.reentrenar()
    modelo = tf.keras.models.load_model("modelo/detector_ia.keras")
    
    texto_resultado.config(
        text="Modelo actualizado exitosamente",
        fg=AcademicStyles.COLORS['success']
    )
    
    barra_confianza.config(text="██████████", fg=AcademicStyles.COLORS['primary_blue'])

# =========================================
# VENTANA PRINCIPAL
# =========================================

ventana = tk.Tk()
ventana.title("Detector IA vs Real - Sistema Académico")
ventana.geometry("1000x750")
ventana.configure(bg=AcademicStyles.COLORS['bg_primary'])
ventana.minsize(900, 700)

# Marco principal
marco_principal = tk.Frame(ventana, bg=AcademicStyles.COLORS['bg_primary'])
marco_principal.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

# Título
titulo_frame = tk.Frame(marco_principal, bg=AcademicStyles.COLORS['bg_primary'])
titulo_frame.pack(fill=tk.X, pady=10)

titulo = tk.Label(
    titulo_frame,
    text="DETECTOR IA vs REAL",
    font=AcademicStyles.FONTS['title'],
    fg=AcademicStyles.COLORS['primary_blue']
)
titulo.pack()

subtitulo = tk.Label(
    titulo_frame,
    text="Sistema de Reconocimiento Neural para Detección de Imágenes",
    font=AcademicStyles.FONTS['subtitle'],
    fg=AcademicStyles.COLORS['text_secondary'],
    bg=AcademicStyles.COLORS['bg_primary']
)
subtitulo.pack()

# Separador decorativo
separador = tk.Frame(marco_principal, height=2, bg=AcademicStyles.COLORS['border'])
separador.pack(fill=tk.X, pady=15)

# Contenido principal
contenedor_principal = tk.Frame(marco_principal, bg=AcademicStyles.COLORS['bg_primary'])
contenedor_principal.pack(fill=tk.BOTH, expand=True, pady=20)

# Lado izquierdo - Imagen
frame_imagen = tk.Frame(contenedor_principal, bg=AcademicStyles.COLORS['bg_primary'], width=500)
frame_imagen.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)
frame_imagen.pack_propagate(False)

# Marco para imagen con borde decorativo
marco_imagen = tk.Frame(frame_imagen, bg=AcademicStyles.COLORS['border'],
                       highlightthickness=2, highlightbackground=AcademicStyles.COLORS['border'],
                       relief=tk.FLAT)
marco_imagen.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

label_imagen = tk.Label(marco_imagen, bg=AcademicStyles.COLORS['bg_secondary'])
label_imagen.pack(expand=True, fill=tk.BOTH, padx=8, pady=8)

# Texto de placeholder
label_imagen.config(
    text="Seleccione una imagen\npara comenzar el análisis",
    font=('Segoe UI', 14),
    fg=AcademicStyles.COLORS['text_tertiary'],
    justify=tk.CENTER
)

# Lado derecho - Controles
frame_derecho = tk.Frame(contenedor_principal, bg=AcademicStyles.COLORS['bg_primary'], width=350)
frame_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
frame_derecho.pack_propagate(False)

# Panel de botones
panel_botones = CardFrame(frame_derecho)
panel_botones.pack(fill=tk.X, pady=10, padx=10)

# Título del panel
tk.Label(
    panel_botones,
    text="CONTROLES",
    font=AcademicStyles.FONTS['subtitle'],
    fg=AcademicStyles.COLORS['text_primary'],
    bg=AcademicStyles.COLORS['bg_secondary']
).pack(pady=15)

# Botón de seleccionar imagen
boton = ProfessionalButton(
    panel_botones,
    text="Seleccionar Imagen",
    command=analizar_imagen,
    font=AcademicStyles.FONTS['button'],
    bg=AcademicStyles.COLORS['primary_blue'],
    fg=AcademicStyles.COLORS['bg_primary']
)
boton.pack(pady=10, padx=20, fill=tk.X)

# Botón de corregir resultado
boton_corregir = SecondaryButton(
    panel_botones,
    text="Corregir Resultado",
    command=corregir,
    font=AcademicStyles.FONTS['button'],
    bg=AcademicStyles.COLORS['bg_secondary'],
    fg=AcademicStyles.COLORS['text_primary']
)
boton_corregir.pack(pady=10, padx=20, fill=tk.X)

# Panel de resultados
panel_resultados = CardFrame(frame_derecho)
panel_resultados.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

tk.Label(
    panel_resultados,
    text="RESULTADOS DEL ANÁLISIS",
    font=AcademicStyles.FONTS['subtitle'],
    fg=AcademicStyles.COLORS['text_primary'],
    bg=AcademicStyles.COLORS['bg_secondary']
).pack(pady=15)

# Barra de confianza
tk.Label(
    panel_resultados,
    text="Nivel de Confianza:",
    font=AcademicStyles.FONTS['body'],
    fg=AcademicStyles.COLORS['text_secondary'],
    bg=AcademicStyles.COLORS['bg_secondary']
).pack(pady=5)

barra_confianza = tk.Label(
    panel_resultados,
    text="██████████",
    font=("Consolas", 18, "bold"),
    fg=AcademicStyles.COLORS['primary_blue'],
    bg=AcademicStyles.COLORS['bg_secondary']
)
barra_confianza.pack(pady=5)

# Resultado
texto_resultado = tk.Label(
    panel_resultados,
    text="Sistema preparado para análisis",
    font=AcademicStyles.FONTS['result'],
    fg=AcademicStyles.COLORS['text_secondary'],
    bg=AcademicStyles.COLORS['bg_secondary'],
    wraplength=300,
    justify=tk.CENTER
)
texto_resultado.pack(pady=20, padx=10)

# Información del modelo
info_frame = tk.Frame(panel_resultados, bg=AcademicStyles.COLORS['bg_secondary'])
info_frame.pack(fill=tk.X, pady=10, padx=10)

tk.Label(
    info_frame,
    text="Información del modelo:",
    font=AcademicStyles.FONTS['small'],
    fg=AcademicStyles.COLORS['text_tertiary'],
    bg=AcademicStyles.COLORS['bg_secondary']
).pack()

tk.Label(
    info_frame,
    text="Modelo: Detector IA v1.0 | TensorFlow",
    font=AcademicStyles.FONTS['small'],
    fg=AcademicStyles.COLORS['text_tertiary'],
    bg=AcademicStyles.COLORS['bg_secondary']
).pack()

# Pie de página
footer = tk.Frame(marco_principal, bg=AcademicStyles.COLORS['bg_primary'])
footer.pack(fill=tk.X, pady=10)

separador2 = tk.Frame(footer, height=1, bg=AcademicStyles.COLORS['border'])
separador2.pack(fill=tk.X, pady=5)

tk.Label(
    footer,
    text="Sistema de Detección Neural | Desarrollado con TensorFlow",
    font=AcademicStyles.FONTS['small'],
    fg=AcademicStyles.COLORS['text_tertiary'],
    bg=AcademicStyles.COLORS['bg_primary']
).pack()

# =========================================
# INICIAR
# =========================================

# Animación de carga inicial
def animacion_carga(index=0):
    textos = ["Iniciando sistema", "Cargando módulos", "Sistema preparado"]
    if index < len(textos):
        texto_resultado.config(text=textos[index])
        ventana.after(500, lambda: animacion_carga(index + 1))
    else:
        texto_resultado.config(text="Sistema preparado para análisis")

animacion_carga()

ventana.mainloop()
