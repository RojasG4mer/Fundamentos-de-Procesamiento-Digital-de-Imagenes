"""
Elaboró: 
ROJAS MARTINEZ JONATHAN FRANCISCO
"""

# librerías
import cv2
import matplotlib.pyplot as plt
import numpy as np

def padding(imagen, dx):
  y, x = imagen.shape
  paddineada = np.zeros((int(y + dx/2 + 1), int(x + dx/2 + 1)))
  for j in range(y):
    for i in range(x):
      paddineada[int(j + dx/2)][int(i + dx/2)] = imagen[j][i]
  return np.array(paddineada)

## Función para recortar la imagen en una zona más pequeña
def Extrae(Im, x, y, dx, dy):
  submatrix = [] #Recorte de la imagen
  fila = []
  temp_col_pix = []
  temp_fila = []
  for j in range(y, y+dy): #nos movemos desde el inicio del corte, hasta el corte
    for i in range(x, x+dx):
      temp_col_pix.append(i)
      fila.append(Im[j][i]) #agregamos el pixel de Im a cada fila nueva
    submatrix.append(fila) #agregamos cada fila a la imagen nueva
    fila = []
    temp_fila.append(temp_col_pix)
    # print(np.array(temp_fila).shape)
    temp_col_pix = []
  return np.array(submatrix)

####### Imagen 

ruta = r'C:\Fundamentos-de-Procesamiento-Digital-de-Imagenes\Imagenes\Lineas_circulo.png'

imagen_rgb = cv2.imread(ruta)
imagen_grises = cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2GRAY)
imagen_grises = Extrae(imagen_grises, 390, 650, 600*2, 678*2)



"""

Filtro de Roberts

"""
roberts_pi4 = np.array([[1, 0], [0, -1]])
roberts_menospi4 = np.array([[0, 1], [-1, 0]])

def ap_filter(filter, imagen):
  y, x = imagen.shape
  sum = 0
  for j in range(y):
    for i in range(x):
      sum += (filter[j][i] * imagen[j][i])
  return sum

def convol(imagen, filter):
  y, x = imagen.shape
  dy, dx = filter.shape
  imagen_nueva = np.zeros((y, x))
  ima_padding = padding(imagen, dy)
  desplaza_y = int(dy/2)
  # desplaza_x = int(dx/2) # Si es cuadrado el filtro no es necesario
  for j in range(y - desplaza_y):
    for i in range(x - desplaza_y):
      imagen_nueva[j][i] = ap_filter(filter, ima_padding[j-desplaza_y:j+desplaza_y, i-desplaza_y:i+desplaza_y])
  return np.array(imagen_nueva)

def opthresholding(imagen, n):
  y, x = imagen.shape
  imagen_nueva = np.zeros((y, x))
  p1 = int(np.max(imagen)/n)
  for j in range(y):
    for i in range(x):
      if imagen[j][i] <= p1:
        imagen[j][i] = 0
      else:
        imagen[j][i] = 255
  return imagen

def mod_grad(grad_x, grad_y, p1):
  y, x = grad_x.shape
  grad_raw_magnitudes = np.zeros((y, x)) # Almacenar las magnitudes antes de la normalización

  for j in range(y):
    for i in range(x):
      # Calcular la norma L1 (suma de los valores absolutos de las derivadas)
      # Esta es una aproximación común para la magnitud del gradiente y cumple con la descripción.
      grad_raw_magnitudes[j][i] = np.abs(grad_x[j][i]) + np.abs(grad_y[j][i])

  # Normalizar todo el array al rango de 0-255
  # Esto evita la saturación y distribuye las magnitudes del gradiente a lo largo
  # de todo el rango de intensidad, evitando que muchos píxeles se saturen a 255.
  min_val = np.min(grad_raw_magnitudes)
  max_val = np.max(grad_raw_magnitudes)

  if max_val == min_val:
    # Si todos los valores son iguales (imagen uniforme), retornar ceros o un valor constante
    grad_mod = np.zeros_like(grad_raw_magnitudes, dtype=np.uint8)
  else:
    # Realizar una normalización min-max para escalar los valores a 0-255
    grad_mod = (grad_raw_magnitudes - min_val) * (255.0 / (max_val - min_val))
    grad_mod = opthresholding(grad_mod, p1).astype(np.uint8) # Convertir a entero sin signo de 8 bits para visualización de imagen

  return grad_mod

n = 3

ima_roberts_pi4 = convol(imagen_grises, roberts_pi4)

ima_roberts_menospi4 = convol(imagen_grises, roberts_menospi4)

ima_grad_roberts = mod_grad(ima_roberts_pi4, ima_roberts_menospi4, n)

fig, axs = plt.subplots(1, 4, figsize = (24, 12))
axs[0].imshow(imagen_grises, cmap = 'gray')
axs[0].set_title('Imagen original')
axs[0].axis('off')

axs[1].imshow(ima_roberts_pi4, cmap = 'gray')
axs[1].set_title(r'Filtro Roberts $\frac{\pi}{4}$')
axs[1].axis('off')

axs[2].imshow(ima_roberts_menospi4, cmap = 'gray')
axs[2].set_title(r'Filtro Roberts $-\frac{\pi}{4}$')
axs[2].axis('off')

axs[3].imshow(ima_grad_roberts, cmap = 'gray')
axs[3].set_title(r'Filtro Roberts $|\nabla f|$')
axs[3].axis('off')

"""

## Filtro de Prewitt

"""

Prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
Prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

ima_prewitt_x = convol(imagen_grises, Prewitt_x)

ima_prewitt_y = convol(imagen_grises, Prewitt_y)

modul_prewitt = mod_grad(ima_prewitt_x, ima_prewitt_y, n)

fig, axs = plt.subplots(1, 4, figsize = (24, 12))
axs[0].imshow(imagen_grises, cmap = 'gray')
axs[0].set_title('Imagen original')
axs[0].axis('off')

axs[1].imshow(ima_prewitt_x, cmap = 'gray')
axs[1].set_title('Filtro Prewitt en x')
axs[1].axis('off')

axs[2].imshow(ima_prewitt_y, cmap = 'gray')
axs[2].set_title('Filtro Prewitt en y')
axs[2].axis('off')

axs[3].imshow(modul_prewitt, cmap = 'gray')
axs[3].set_title(R'Filtro Prewitt $|\nabla f|$')
axs[3].axis('off')

"""

Filtro de Sobel


"""

Sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Sobel_y = np.array([[-1, -2, 1], [0, 0, 0], [-1, 2, 1]])

ima_sobel_x = convol(imagen_grises, Sobel_x)

ima_sobel_y = convol(imagen_grises, Sobel_y)

modul_Sobel = mod_grad(ima_sobel_x, ima_sobel_y, n)

fig, axs = plt.subplots(1, 4, figsize = (24, 12))
axs[0].imshow(imagen_grises, cmap = 'gray')
axs[0].set_title('Imagen original')
axs[0].axis('off')

axs[1].imshow(ima_sobel_x, cmap = 'gray')
axs[1].set_title('Filtro Sobel en x')
axs[1].axis('off')

axs[2].imshow(ima_sobel_y, cmap = 'gray')
axs[2].set_title('Filtro Sobel en y')
axs[2].axis('off')

axs[3].imshow(modul_Sobel, cmap = 'gray')
axs[3].set_title(R'Filtro Sobel $|\nabla f|$')
axs[3].axis('off')

"""

Operador de Kirsch

"""

D1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
D2 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
D3 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
D4 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])

# Calculamos las direcciones

im_D1 = convol(imagen_grises, D1)

im_D2 = convol(imagen_grises, D2)

im_D3 = convol(imagen_grises, D3)

im_D4 = convol(imagen_grises, D4)

# Seleccionamos el más grande de los 4 para cada pixel
def operador_kirsch(imagen):
  y, x = imagen.shape
  imagen_kirsch = np.zeros((y, x))
  for j in range(y):
    for i in range(x):
      arr_mag = np.array([im_D1[j][i], im_D2[j][i], im_D3[j][i], im_D4[j][i]])
      imagen_kirsch[j][i] = np.max(arr_mag)
  p1 = int(np.max(imagen_kirsch)/3)
  print(p1)
  imagen_kirsch = np.array(opthresholding(imagen_kirsch, p1))
  return imagen_kirsch

imagen_kirsch = operador_kirsch(imagen_grises)

fig, axs = plt.subplots(1, 2, figsize = (12, 6))
axs[0].imshow(imagen_grises, cmap = 'gray')
axs[0].set_title('Imagen original')
axs[0].axis('off')

axs[1].imshow(imagen_kirsch, cmap = 'gray')
axs[1].set_title('Operador Kirsch (máximo)')
axs[1].axis('off')

def operador_kirsch(imagen):
    # Definición de los 8 kernels de brújula de Kirsch
    k1 = np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]]) # Norte
    k2 = np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]]) # Noroeste
    k3 = np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]]) # Oeste
    k4 = np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]]) # Suroeste
    k5 = np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]]) # Sur
    k6 = np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]]) # Sureste
    k7 = np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]]) # Este
    k8 = np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]]) # Noreste

    kernels = [k1, k2, k3, k4, k5, k6, k7, k8]

    # Crear un arreglo tridimensional para guardar las 8 respuestas
    y, x = imagen.shape
    respuestas = np.zeros((8, y, x))

    # Aplicar cada kernel a la imagen
    for i, k in enumerate(kernels):
        # cv2.filter2D es computacionalmente equivalente a tu función convol
        respuestas[i] = cv2.filter2D(imagen, cv2.CV_64F, k)

    # 1. Magnitud del gradiente: el máximo valor absoluto en cada píxel
    # G(x, y) = max{|D_1|, |D_2|, ..., |D_8|}
    magnitud_kirsch = np.max(np.abs(respuestas), axis=0)

    # Normalizar a 0-255 como en tu función mod_grad
    min_val = np.min(magnitud_kirsch)
    max_val = np.max(magnitud_kirsch)

    if max_val != min_val:
        magnitud_kirsch = (magnitud_kirsch - min_val) * (255.0 / (max_val - min_val))

    return magnitud_kirsch.astype(np.uint8)

# --- Ejecución y Visualización ---
# Asegúrate de tener 'imagen_grises' cargada en tu entorno antes de ejecutar esto
ima_kirsch = operador_kirsch(imagen_grises)

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
axs[0].imshow(imagen_grises, cmap='gray')
axs[0].set_title('Imagen original')
axs[0].axis('off')

axs[1].imshow(ima_kirsch, cmap='gray')
axs[1].set_title('Operador de Kirsch (Magnitud Máxima)')
axs[1].axis('off')

plt.show()