import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. FUNCIONES AUXILIARES (Puro Python)
# ==========================================

def crear_matriz(alto, ancho, valor):
    """Crea una matriz (lista de listas) llena de un valor específico."""
    return [[valor for _ in range(ancho)] for _ in range(alto)]

def copiar_matriz(matriz):
    """Crea una copia independiente de una matriz."""
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

# ==========================================
# 1. FUNCIONES MORFOLÓGICAS (Puro Python)
# ==========================================

def dilatar(imagen, kernel):
    """Dilatación usando solo bucles y listas."""
    alto_img, ancho_img = len(imagen), len(imagen[0])
    alto_k, ancho_k = len(kernel), len(kernel[0])
    
    pad_h = alto_k // 2
    pad_w = ancho_k // 2
    
    # Añadimos borde de 0s
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

def cerrar_circulos(imagen, kernel):
    """Cierre morfológico: Dilatación seguida de Erosión."""
    img_dilatada = dilatar(imagen, kernel)
    img_cerrada = erosionar(img_dilatada, kernel)
    return img_cerrada

# ===== FUNCIONES DE LÓGICA PARA EL ESQUELETO =====
def es_matriz_vacia(matriz):
    """Revisa si todos los pixeles de la matriz son 0."""
    for fila in matriz:
        for valor in fila:
            if valor != 0:
                return False # Encontró un 1, no está vacía
    return True

def operacion_logica(img1, img2, operacion):
    """Realiza operaciones AND_NOT y OR píxel por píxel."""
    alto, ancho = len(img1), len(img1[0])
    resultado = crear_matriz(alto, ancho, 0)
    for i in range(alto):
        for j in range(ancho):
            if operacion == "AND_NOT":
                # Si img1 es 1 y img2 es 0 -> Resultado 1
                if img1[i][j] == 1 and img2[i][j] == 0:
                    resultado[i][j] = 1
            elif operacion == "OR":
                # Si cualquiera es 1 -> Resultado 1
                if img1[i][j] == 1 or img2[i][j] == 1:
                    resultado[i][j] = 1
    return resultado

def obtener_esqueleto(imagen, kernel):
    """Esqueleto usando iteración de erosiones puramente en Python."""
    alto, ancho = len(imagen), len(imagen[0])
    esqueleto = crear_matriz(alto, ancho, 0)
    img_temp = copiar_matriz(imagen)
    
    while not es_matriz_vacia(img_temp):
        # 1. Erosionamos
        erosionada = erosionar(img_temp, kernel)
        
        # 2. Apertura (Dilatamos la recién erosionada)
        abierta = dilatar(erosionada, kernel)
        
        # 3. Residuo = img_temp AND (NOT abierta)
        residuo = operacion_logica(img_temp, abierta, "AND_NOT")
        
        # 4. Acumulamos en el esqueleto: esqueleto = esqueleto OR residuo
        esqueleto = operacion_logica(esqueleto, residuo, "OR")
        
        # 5. La imagen para la próxima iteración es la erosionada
        img_temp = copiar_matriz(erosionada)
        
    return esqueleto





# =========================================================================================================================================================



def dilate_from_scratch(image, kernel_size=3):
    # Get image dimensions
    h, w = image.shape
    # Calculate padding based on kernel size (assumes odd kernel)
    pad = kernel_size // 2
    # Create an output image of the same size, initialized to 0
    dilated_img = np.zeros_like(image)
    
    # Pad the original image to handle borders
    padded_img = np.pad(image, pad, mode='constant', constant_values=0)
    
    # Iterate through every pixel of the original image
    for i in range(h):
        for j in range(w):
            # Extract the neighborhood region (window)
            window = padded_img[i : i + kernel_size, j : j + kernel_size]
            
            # The output pixel is the maximum of the neighborhood
            # In binary images, if any pixel is 1, max() will return 1
            dilated_img[i, j] = np.max(window)
            
    return dilated_img

# Example Usage:
# binary_mask = (your_grayscale_image > threshold).astype(np.uint8)
# result = dilate_from_scratch(binary_mask, kernel_size=5)



# ==========================================
# 2. CARGA, EJECUCIÓN Y CONVERSIÓN A UINT8
# ==========================================

# Cargamos la imagen con OpenCV 
ruta = r'Temas\04_Procesamiento_morfologico\Copia de Casa con bordes.jpg'
img_original = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
img_original = img_original[250:300, 150:200]


# Binarizamos y obtenemos un arreglo de NumPy
# _, img_binaria_np = cv2.threshold(img_original, 127, 1, cv2.THRESH_BINARY)
_, img_binaria_np = cv2.threshold(img_original, 127, 1, cv2.THRESH_BINARY_INV)


# CONVERSIÓN CRÍTICA: Pasamos la matriz de OpenCV/NumPy a listas puras de Python
img_lista_pura = img_binaria_np.tolist()

# Definimos un kernel de 3x3 como lista pura de Python
kernel_lista = [
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
]

# kernel_lista = [
#     [0, 0, 1],
#     [0, 1, 1],
#     [1, 1, 1]
# ]


# ¡Procesamos TODO usando las listas puras de Python!
print("Calculando Erosión...")
res_erosionada = erosionar(img_lista_pura, kernel_lista)

print("Calculando Dilatación...")
res_dilatada = dilatar(img_lista_pura, kernel_lista)

print("Calculando Cierre...")
res_cerrada = cerrar_circulos(img_lista_pura, kernel_lista)

print("Calculando Esqueleto...")
res_esqueleto = obtener_esqueleto(img_lista_pura, kernel_lista)

# ==========================================
# 3. VISUALIZACIÓN DE RESULTADOS
# ==========================================

# Aquí es el ÚNICO lugar donde usamos numpy. 
# Envolvemos las listas de Python con np.array(..., dtype=np.uint8) 
# y multiplicamos por 255 para que Matplotlib lo dibuje en blanco y negro estándar.

titulos = ['Original (Binaria)', 'Erosión', 'Dilatación', 'Cierre (Tapa huecos)', 'Esqueleto']

imagenes = [
    np.array(img_lista_pura, dtype=np.uint8) * 255,
    np.array(res_erosionada, dtype=np.uint8) * 255,
    np.array(res_dilatada, dtype=np.uint8) * 255,
    np.array(res_cerrada, dtype=np.uint8) * 255,
    np.array(res_esqueleto, dtype=np.uint8) * 255
]

plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(imagenes[i], cmap='gray')
    plt.title(titulos[i])
    plt.axis('off')

plt.tight_layout()
plt.show()