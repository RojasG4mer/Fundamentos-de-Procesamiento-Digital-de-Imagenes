import cv2
import matplotlib.pyplot as plt
import numpy as np

ruta = r'Imagenes/Copia de Casa con bordes.jpg' # Casa con bordes

ima = cv2.imread(ruta)
ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
ima = ima[200:300, 150:350]
filas, columnas = ima.shape

# plt.imshow(ima, cmap = 'gray')

# print(ima.shape)

#### Transformada de Fourier Discreta 2D

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

# Centramos la imagen (quede como en el libro)
transformada_centrada = np.fft.fftshift(transformada_manual)
# Normalizamos el espectro para visualizarlo mejor (logaritmo para reducir rango dinámico)
espectro_magnitud = np.log(np.abs(transformada_centrada) + 1)

# # Visualizamos el resultado
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(ima, cmap='gray')
# plt.title('Imagen Original $f(x, y)$')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(espectro_magnitud, cmap='gray')
# plt.title('Espectro Manual $F(u, v)$')
# plt.axis('off')
# plt.show()


##### Aplicar un filtro

# Máscara Pasa-Bajas
# Calculamos el centro del filtro (en el espacio de las frecuencias)
centro_fila, centro_columna = filas // 2, columnas // 2 
radio_de_corte = 10 # Qué tanto queremos difuminar (menor radio = más borroso)

# Creamos la máscara llena de ceros
mascara = np.zeros((filas, columnas), np.uint8)

# Calculamos la distancia de cada píxel al centro para dibujar el círculo
mascara = np.zeros((filas, columnas))

for y in range(filas):
    for x in range(columnas):
        # Distancia
        distancia = np.sqrt((x - centro_columna)**2 + (y - centro_fila)**2)
        # Verificamos si es menor o mayor al corte
        if distancia <= radio_de_corte:
            mascara[y, x] = 1

Filtrada = np.copy(transformada_manual)

# Aplicar el filtro
for j in range(filas):
    for i in range(columnas):
        Filtrada[j, i] *= mascara[j, i]


# Antitransformada:
def Antift_2d_manual(imagen):
    """
    Calcula la Transformada de Fourier Discreta 2D paso a paso según la fórmula matemática.
    """
    M, N = imagen.shape

    # Debe ser de tipo complex porque la fórmula usa números imaginarios.
    F = np.zeros((M, N), dtype=complex)

    # Iteramos sobre cada frecuencia u (eje vertical) y v (eje horizontal)
    for u in range(M):
        for v in range(N):
            suma_compleja = 0
            # Iteramos sobre cada píxel de la imagen original f(x,y)
            for x in range(M):
                for y in range(N):

                    # Obtenemos el valor del píxel en la posición (x, y)
                    f_xy = imagen[x, y]

                    # Calculamos el exponente
                    # Nota: En Python, el número imaginario 'j' se escribe como '1j'
                    exponente = 1j * 2 * np.pi * ((u * x) / M + (v * y) / N)

                    # Calculamos f(x,y) * e^(exponente) y lo añadimos a la sumatoria
                    suma_compleja += f_xy * np.exp(exponente)

            # Guardamos el resultado final de las sumatorias en F(u, v)
            F[u, v] = suma_compleja

    return (F) / (N * M)

anti = np.abs(Antift_2d_manual(Filtrada))


# Filtrada = np.fft.fftshift(Filtrada)
# Normalizamos el espectro para visualizarlo mejor (logaritmo para reducir rango dinámico)
Filtrada = np.log(np.abs(Filtrada) + 1)

# Graficamos 
fig, axs = plt.subplots(1, 4, figsize=(24, 6))

axs[0].imshow(ima, cmap='gray')
axs[0].set_title('Imagen Original $f(x, y)$')  
axs[0].axis('off')

axs[1].imshow(mascara, cmap='gray')
axs[1].set_title(f'Filtro $F(u, v)$, con radio de corte {radio_de_corte}')
axs[1].axis('off')

axs[2].imshow(Filtrada, cmap='gray')
axs[2].set_title(r'Transformada de Fourier Filtrada $F(u, v) \cdot H(u, v)$')
axs[2].axis('off')

axs[3].imshow(anti, cmap='gray')
axs[3].set_title('Imagen Filtrada (Antitransformada)')
axs[3].axis('off')

plt.show()
