import cv2
import numpy as np
import matplotlib.pyplot as plt

def crear_matriz(alto, ancho, valor):
    return [[valor for _ in range(ancho)] for _ in range(alto)]

def copiar_matriz(matriz):
    return [fila[:] for fila in matriz]

def agregar_borde(imagen, pad_h, pad_w, valor_borde):
    """Añade un borde a la imagen para que el kernel no se salga de los límites."""
    alto = len(imagen)
    ancho = len(imagen[0])
    # Calculamos el nuevo tamaño
    nuevo_alto = alto + 2 * pad_h
    nuevo_ancho = ancho + 2 * pad_w
    
    # Creamos un lienzo más grande lleno del valor del borde
    matriz_borde = crear_matriz(nuevo_alto, nuevo_ancho, valor_borde)
    
    # Copiamos la imagen original en el centro de este lienzo
    for i in range(alto):
        for j in range(ancho):
            matriz_borde[i + pad_h][j + pad_w] = imagen[i][j]
            
    return matriz_borde

# Funciones morfologicas básicas ----------------------------------------------------------------------------------------------------

def dilatar(imagen, kernel):

    alto_img, ancho_img = len(imagen), len(imagen[0])
    alto_k, ancho_k = len(kernel), len(kernel[0])
    
    pad_h = alto_k // 2
    pad_w = ancho_k // 2
    
    # Añadimos padding
    img_borde = agregar_borde(imagen, pad_h, pad_w, 0)
    resultado = crear_matriz(alto_img, ancho_img, 0)
    
    for i in range(alto_img):
        for j in range(ancho_img):
            # Superponemos el kernel sobre la imagen original
            coincide = False
            for ki in range(alto_k):
                for kj in range(ancho_k):
                    # Si el kernel es 1 y el pixel de abajo de la imagen también es 1, hay coincidencia
                    if kernel[ki][kj] == 1 and img_borde[i + ki][j + kj] == 1:
                        coincide = True
                        break # Si ya encontramos uno, no hace falta buscar más en esta ventana
                if coincide:
                    break
            
            # Si hubo al menos una coincidencia, el pixel central se vuelve 1
            if coincide:
                resultado[i][j] = 1
                
    return resultado

def erosionar(imagen, kernel):
    """Erosión usando solo bucles y listas."""
    alto_img, ancho_img = len(imagen), len(imagen[0])
    alto_k, ancho_k = len(kernel), len(kernel[0])
    
    pad_h = alto_k // 2
    pad_w = ancho_k // 2
    
    # Para erosión el borde DEBE ser 1 para no afectar falsamente los bordes reales de la imagen
    img_borde = agregar_borde(imagen, pad_h, pad_w, 1)
    resultado = crear_matriz(alto_img, ancho_img, 0)
    
    for i in range(alto_img):
        for j in range(ancho_img):
            # Asumimos que todo va a coincidir
            coincide_todo = True
            for ki in range(alto_k):
                for kj in range(ancho_k):
                    # Si el kernel exige un 1, pero la imagen tiene un 0, se rompe la erosión
                    if kernel[ki][kj] == 1 and img_borde[i + ki][j + kj] == 0:
                        coincide_todo = False
                        break
                if not coincide_todo:
                    break
            
            # Solo si TODOS los pixeles bajo el kernel coincidieron, se marca como 1
            if coincide_todo:
                resultado[i][j] = 1
                
    return resultado


# ----------------------------------------------------------------------------------------------------

# Cargamos la 
ruta = r'Temas\04_Procesamiento_morfologico\Copia de Casa con bordes.jpg'
img_original = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
img_original = img_original[250:300, 150:200]


# Binarizamos y hacemos que se invierta para el fondo negro y objeto blanco
# _, img_binaria_np = cv2.threshold(img_original, 127, 1, cv2.THRESH_BINARY)
_, img_binaria_np = cv2.threshold(img_original, 127, 1, cv2.THRESH_BINARY_INV)


# CONVERSIÓN CRÍTICA: Pasamos la matriz de OpenCV/NumPy a listas puras de Python
img_lista_pura = img_binaria_np.tolist()

# Definimos un kernel de 3x3 como lista pura de Python
kernel_lista = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
]

# kernel_lista = [
#     [0, 0, 1],
#     [0, 1, 1],
#     [1, 1, 1]
# ]

# Aplicar funciones
res_erosionada = erosionar(img_lista_pura, kernel_lista)

res_dilatada = dilatar(img_lista_pura, kernel_lista)

# ==========================================
# ---------- Resultados
# ==========================================

# Multiplicamos por 255 para que Matplotlib lo dibuje en blanco y negro estándar.

titulos = ['Original (Binaria)', 'Erosión', 'Dilatación']

imagenes = [
    np.array(img_lista_pura, dtype=np.uint8) * 255,
    np.array(res_erosionada, dtype=np.uint8) * 255,
    np.array(res_dilatada, dtype=np.uint8) * 255,
]

plt.figure(figsize=(15, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(imagenes[i], cmap='gray')
    plt.title(titulos[i])
    plt.axis('off')

plt.tight_layout()
plt.show()