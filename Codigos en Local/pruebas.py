import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar la imagen en escala de grises
# (Asegúrate de poner la ruta correcta a tu archivo)
imagen = cv2.imread(r'Imagenes\Copia de Casa con bordes.jpg', 0)
filas, columnas = imagen.shape

# 2. Transformada de Fourier (Viaje al dominio de la frecuencia)
f = np.fft.fft2(imagen)
fshift = np.fft.fftshift(f) # Centramos las bajas frecuencias

# 3. Crear el Filtro (Máscara Pasa-Bajas)
# Haremos una matriz de ceros (negro) y pondremos un círculo de unos (blanco) en el centro
centro_fila, centro_columna = filas // 2, columnas // 2
radio_de_corte = 30 # Qué tanto queremos difuminar (menor radio = más borroso)

# Creamos la máscara llena de ceros
mascara = np.zeros((filas, columnas), np.uint8)

# Calculamos la distancia de cada píxel al centro para dibujar el círculo
Y, X = np.ogrid[:filas, :columnas]
distancia_al_centro = np.sqrt((X - centro_columna)**2 + (Y - centro_fila)**2)

# Rellenamos el círculo central con 1s
mascara[distancia_al_centro <= radio_de_corte] = 1

# 4. Aplicar el Filtro
# Multiplicamos el espectro centrado por nuestra máscara
espectro_filtrado = fshift * mascara

# 5. Transformada Inversa (Regreso al dominio espacial)
# Primero "des-centramos" el espectro
f_ishift = np.fft.ifftshift(espectro_filtrado)

# Aplicamos la Transformada Inversa (IFFT)
imagen_inversa = np.fft.ifft2(f_ishift)

# El resultado son números complejos, así que tomamos el valor absoluto (magnitud)
imagen_filtrada = np.abs(imagen_inversa)

# --- Visualización de los Resultados ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 3, 2)
# Visualizamos la máscara para que veas qué forma tiene
plt.imshow(mascara, cmap='gray')
plt.title(f'Máscara Pasa-Bajas (Radio {radio_de_corte})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(imagen_filtrada, cmap='gray')
plt.title('Imagen Filtrada (Resultado)')
plt.axis('off')

plt.tight_layout()
plt.show()