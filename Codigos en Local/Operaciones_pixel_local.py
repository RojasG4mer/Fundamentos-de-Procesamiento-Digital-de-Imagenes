# -*- coding: utf-8 -*-
"""
Operaciones a nivel pixel - Versión Local
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# CONFIGURACIÓN DE LA IMAGEN
# ==========================================
# Cambia esto por la ruta de tu imagen o pon la imagen en la misma carpeta que este script
nombre_imagen = r'C:\Fundamentos-de-Procesamiento-Digital-de-Imagenes\figuras.BMP' 
# Popi_fachero.jpg
# Verificamos si existe el archivo antes de intentar leerlo
if not os.path.exists(nombre_imagen):
    print(f"ERROR CRÍTICO: No se encuentra el archivo '{nombre_imagen}' en la ruta actual.")
    print(f"Ruta actual de trabajo: {os.getcwd()}")
    exit()

# ==========================================
# 1. LECTURA Y VISUALIZACIÓN INICIAL
# ==========================================

# Leer la imagen (OpenCV lee en BGR)
popi_fachero = cv2.imread(nombre_imagen)

if popi_fachero is None:
    print("Error: No se pudo cargar la imagen. El formato podría ser incorrecto.")
    exit()

# Convertir de BGR a RGB para mostrarla correctamente
popi_fachero_rgb = cv2.cvtColor(popi_fachero, cv2.COLOR_BGR2RGB)

# Mostrar la imagen original
plt.figure(figsize=(8, 6))
plt.imshow(popi_fachero_rgb)
plt.title('Imagen Original')
plt.axis('off')
plt.show()

# Convertir a escala de grises
popi_fachero_grises = cv2.cvtColor(popi_fachero_rgb, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 6))
plt.imshow(popi_fachero_grises, cmap='gray')
plt.title('Imagen en Grises')
plt.axis('off')
plt.show()

# ==========================================
# 2. HISTOGRAMA
# ==========================================

# Definimos la función de histograma artesanal aquí para usarla más adelante
def histo_artesanal(imagen):
    # Aseguramos que sea array de numpy para .max() y .min()
    
    imagen = np.array(imagen) 
    max_val = imagen.max()
    min_val = imagen.min()
    # Creamos arreglo de ceros (usualmente 256 espacios para 0-255)
    histograma = [0] * 256 
    
    for fila in imagen:
        for pixel in fila:
            histograma[pixel] += 1 
    return np.array(histograma)

# Cálculo del histograma
histograma = histo_artesanal(popi_fachero_grises)

# Cálculo con OpenCV para comparar
hist_cv2 = cv2.calcHist([popi_fachero_grises], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 5))
plt.bar(range(len(histograma)), histograma, color='r', alpha=0.5, label='Manual')
plt.plot(hist_cv2, color='g', label='OpenCV')
plt.title('Comparación de Histogramas')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ==========================================
# 3. OPERACIONES A NIVEL PÍXEL
# ==========================================

# Función auxiliar para graficar resultados
def grafi_imag(imagen, titulo):
    # Convertimos a numpy array uint8 para que matplotlib no se queje
    img_np = np.array(imagen, dtype=np.uint8)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_np, cmap='gray', vmin=0, vmax=255) # Forzamos escala 0-255
    plt.title(titulo)
    plt.axis('off')
    plt.show()

print("Procesando operaciones a nivel píxel (esto puede tardar unos segundos)...")

# --- Operador Identidad ---
# (Copia exacta)
popi_fachero_identidad = popi_fachero_grises.copy() # Es más rápido usar copy() de numpy
grafi_imag(popi_fachero_identidad, 'Operador Identidad')

# --- Operador Identidad Inverso (Negativo) ---
def opidentidadinverso(p):
    return 255 - p

# Usamos listas por comprensión para mantener la lógica manual del original, 
# aunque vectorizado con numpy sería instantáneo: img = 255 - img
popi_fachero_inverso = [[opidentidadinverso(p) for p in fila] for fila in popi_fachero_grises]
grafi_imag(popi_fachero_inverso, 'Identidad Inverso (Negativo)')

# --- Operador Umbral (Thresholding) ---
def opthresholding(p, p1):
    return 0 if p <= p1 else 255

popi_fachero_umbral = [[opthresholding(p, 100) for p in fila] for fila in popi_fachero_grises]
grafi_imag(popi_fachero_umbral, 'Umbral Binario (p1=100)')

# --- Operador Umbral Escala de Grises ---
# Mantiene grises en rango, el resto a blanco
def opthresholding_escalagrises(p, p1, p2):
    if p1 <= p <= p2:
        return p
    else:
        return 255

popi_fachero_umbral_grises = [[opthresholding_escalagrises(p, 100, 200) for p in fila] for fila in popi_fachero_grises]
grafi_imag(popi_fachero_umbral_grises, 'Umbral Escala de Grises (100-200)')

# --- Operador Reducción de Niveles ---
def opthresholding_red_escala_grises(p, p_escalas, q_valores):
    # Nota: Es importante que p_escalas esté ordenado
    for i, escala in enumerate(p_escalas):
        if p <= escala:
            return q_valores[i]
    return q_valores[-1]

p_escalas = [100, 150, 200, 240]
q_valores = [50, 100, 150, 250]
# Ordenamos por seguridad
p_escalas.sort()
q_valores.sort()

popi_fachero_red_grises = [[opthresholding_red_escala_grises(p, p_escalas, q_valores) for p in fila] for fila in popi_fachero_grises]
grafi_imag(popi_fachero_red_grises, 'Reducción de Niveles de Gris')

# ==========================================
# 4. MÉTODO DE OTSU
# ==========================================

print("Calculando Otsu...")

# Implementación OpenCV
otsu_val_cv2, img_otsu_cv2 = cv2.threshold(popi_fachero_grises, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f'Umbral calculado por OpenCV (Otsu): {otsu_val_cv2}')

# Implementación Manual
def manual_otsu(image):
    hist = histo_artesanal(image)
    hist_norm = hist / hist.sum() # Probabilidad de cada nivel

    cum_sum = np.cumsum(hist_norm) # Probabilidad acumulada (omega)
    # Media acumulada (mu)
    cum_mean = np.cumsum(hist_norm * np.arange(len(hist))) 

    global_mean = cum_mean[-1] # Media global

    max_variance = 0
    best_threshold = 0

    # Iterar sobre todos los posibles umbrales (0 a 255)
    for t in range(256):
        w0 = cum_sum[t]
        w1 = 1.0 - w0

        if w0 > 0 and w1 > 0:
            mu0 = cum_mean[t] / w0
            mu1 = (global_mean - cum_mean[t]) / w1
            
            # Varianza entre clases (between-class variance)
            variance = w0 * w1 * ((mu0 - mu1) ** 2)

            if variance > max_variance:
                max_variance = variance
                best_threshold = t

    return best_threshold

umbral_manual = manual_otsu(popi_fachero_grises)
print(f'Umbral calculado manualmente: {umbral_manual}')

# Aplicar el umbral manual
img_otsu_manual = (popi_fachero_grises > umbral_manual).astype(np.uint8) * 255

# Comparación Visual Otsu
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_otsu_cv2, cmap='gray')
plt.title(f'Otsu OpenCV ({otsu_val_cv2})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_otsu_manual, cmap='gray')
plt.title(f'Otsu Manual ({umbral_manual})')
plt.axis('off')
plt.show()

# Graficar umbral sobre histograma
plt.figure(figsize=(8, 4))
plt.bar(range(len(histograma)), histograma, color='gray', alpha=0.7)
plt.axvline(x=umbral_manual, color='r', linestyle='--', linewidth=2, label=f'Umbral Otsu ({umbral_manual})')
plt.title('Histograma y Umbral Otsu')
plt.legend()
plt.show()

# ==========================================
# 5. RECONOCIMIENTO POR LÓBULOS
# ==========================================
# Basado en la observación del histograma del código original:
# Zonas: (100, 160) y (180, 250)

print("Generando segmentación por lóbulos...")

lobulo1 = [[opthresholding_escalagrises(p, 100, 160) for p in fila] for fila in popi_fachero_grises]
lobulo2 = [[opthresholding_escalagrises(p, 180, 250) for p in fila] for fila in popi_fachero_grises]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.array(lobulo1, dtype=np.uint8), cmap='gray', vmin=0, vmax=255)
plt.title('Primer Lóbulo (100-160)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.array(lobulo2, dtype=np.uint8), cmap='gray', vmin=0, vmax=255)
plt.title('Segundo Lóbulo (180-250)')
plt.axis('off')
plt.show()

print("Ejecución finalizada.")