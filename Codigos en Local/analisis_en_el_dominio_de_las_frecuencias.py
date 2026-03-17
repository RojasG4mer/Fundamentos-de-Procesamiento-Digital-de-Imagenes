import cv2
import matplotlib.pyplot as plt
import numpy as np

ruta = r'Imagenes/Casa con bordes.tif' # Casa con bordes
# '/content/drive/MyDrive/Procesamiento Digital del Imágenes/Imagenes/figuras.BMP' # Figuras
ima = cv2.imread(ruta)
ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
ima = ima[200:300, 150:350]
plt.imshow(ima, cmap = 'gray')

print(ima.shape)

def dft_2d_manual(imagen):
    """
    Calcula la Transformada de Fourier Discreta 2D paso a paso según la fórmula matemática.
    """
    # 1. Obtenemos las dimensiones de la imagen: M (filas) y N (columnas)
    M, N = imagen.shape

    # 2. Creamos una matriz vacía para guardar los resultados F(u, v).
    # Debe ser de tipo complejo (complex) porque la fórmula usa números imaginarios.
    F = np.zeros((M, N), dtype=complex)

    # 3. Iteramos sobre cada frecuencia u (eje vertical) y v (eje horizontal)
    for u in range(M):
        for v in range(N):

            suma_compleja = 0

            # 4. Iteramos sobre cada píxel de la imagen original f(x,y)
            for x in range(M):
                for y in range(N):

                    # Obtenemos el valor del píxel en la posición (x, y)
                    f_xy = imagen[x, y]

                    # Calculamos el exponente: -j * 2 * pi * (ux/M + vy/N)
                    # Nota: En Python, el número imaginario 'j' se escribe como '1j'
                    exponente = -1j * 2 * np.pi * ((u * x) / M + (v * y) / N)

                    # Calculamos f(x,y) * e^(exponente) y lo añadimos a la sumatoria
                    suma_compleja += f_xy * np.exp(exponente)

            # 5. Guardamos el resultado final de las sumatorias en F(u, v)
            F[u, v] = suma_compleja

    return F

transformada_manual = dft_2d_manual(ima)

# Aunque hicimos la DFT manual, usaremos estas dos funciones para centrarla
# y prepararla visualmente, tal como lo hicimos en el ejemplo anterior.
transformada_centrada = np.fft.fftshift(transformada_manual)
espectro_magnitud = np.log(np.abs(transformada_centrada) + 1)

# Visualizamos el resultado
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(ima, cmap='gray')
plt.title('Imagen Original $f(x, y)$')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(espectro_magnitud, cmap='gray')
plt.title('Espectro Manual $F(u, v)$')
plt.axis('off')
plt.show()


##### Aplicar un filtro






