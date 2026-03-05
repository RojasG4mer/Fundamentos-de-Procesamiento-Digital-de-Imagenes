import cv2
import matplotlib.pyplot as plt
import numpy as np

# Importar la imagen de uso
ruta = r'C:\Fundamentos-de-Procesamiento-Digital-de-Imagenes\Imagenes\bebida.jpg'
# ruta = f'/content/drive/MyDrive/Procesamiento Digital del Imágenes/Imagenes/gato_nojao.jpg'
imagen = cv2.imread(ruta)


# Ya que OpenCV lee las imágenes como BGR, necesitamos cambiar el orden de los valores para poder mostrar la imagen en el formato correcto
imagen_BGR = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
pixels = np.array(imagen_BGR)

# TAmbién se puede hacer la inversión usando slicing
# image = cv2.imread("path/to/file/image.jpg")[:,:,::-1]

def histo_casero(imagen):
  # Convertimos la imagen a uint8 para asegurar que los valores sean índices válidos (0-255)
  img_int = imagen.astype(np.uint8)
  histo = [0]*256
  for fila in img_int:
    for pixel in fila:
      histo[pixel] += 1
  return np.array(histo).astype(np.int64)

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
histo_imagen = histo_casero(imagen_gris)


"""######## Operador de extensión ######## """

def op_extension(imagen, p1, p2):
  imagen_extendida = np.zeros(imagen.shape)
  mask_medio = (imagen >= p1) & (imagen <= p2) # Solo necesitamos los de en medio, porque los demás son cero
  imagen_extendida[mask_medio] = (imagen[mask_medio] - p1) * (255 / (p2 - p1)) # aplicamos la fórmula
  return imagen_extendida

# bebida
p1 = 50 #50
p2 = 100
#gato nojao
# p1 = 0
# p2 = 50

imagen_extendida = op_extension(imagen_gris, p1, p2)
histo_extendido = histo_casero(imagen_extendida)

"""######## Función de escalamiento de intensidad paramétrica ######## """

def lambda_tans(alpha_an, beta_an):
  return np.tan(beta_an) / np.tan(alpha_an)

"""### Factor de escala base #### """

def xi_cte(imagen, f_u, f_1, lamb):
  # Convertimos a float para evitar que el cálculo se limite a 255 (uint8)
  f_max = float(np.max(imagen))
  f_u = float(f_u)
  f_1 = float(f_1)

  denominador = f_max + (f_u - f_1) * (lamb - 1)
  resultado = f_max / denominador
  print(f"Factor xi calculado: {resultado}")
  return resultado

"""#### Punto de control inferior $f'_{1}$####"""

def f_1pr(xi, f_1):
  return xi * f_1

"""### Punto de control superior $f_u'$####"""

def f_upr(xi, lam, f_u, f_1):
  return xi*(lam * (f_u - f_1) + f_1)

def inte_param(imagen, f_u, f_1, lam):
  xi = xi_cte(imagen, f_u, f_1, lam)
  f_1prim = f_1pr(xi, f_1)
  f_uprim = f_upr(xi, lam, f_u, f_1)
  im_salida = np.zeros(imagen.shape)
  # Máscaras de los valores
  mask_baja = imagen <= f_1
  mask_media = (imagen > f_1) & (imagen < f_u)
  mask_alta = imagen >= f_u
  # Aplicamos las máscaras con los valores correspondientes:
  im_salida[mask_baja] = imagen[mask_baja] * xi
  im_salida[mask_media] = xi * lam * (imagen[mask_media] - f_1) + f_1prim
  im_salida[mask_alta] = xi * (imagen[mask_alta] - f_u) + f_uprim
  return im_salida

# parámetros
lam = 1
fu = 50
f1 = 100

#bebida
ima_escala = inte_param(imagen_gris, f1, fu, lam)
# gato nojao
# ima_escala = inte_param(imagen_gris, 0, 50, -20)
histo_escala = histo_casero(ima_escala)


"""######## Ecualización de una imagen ########"""

def FDP(imagen):
  histograma = histo_casero(imagen)
  total = np.sum(histograma)
  fdp = []
  for i in histograma:
    fdp.append(i/total)
  return fdp, histograma

def FAP(imagen):
  fdp, histo = FDP(imagen)
  fap = [0]*len(histo)
  TEMP = 0
  for i in range(len(histo)):
    fap[i] = fdp[i] + TEMP
    TEMP = fap[i]
  return fap, histo

def T_rk(imagen):
  fap, histo = FAP(imagen)
  trk = []
  N = np.max(imagen)
  for i in fap:
    trk.append(int((N-1) * i))
  return trk, histo

def ecualizacion(imagen):
  fdp_vals, histo = FDP(imagen)
  fap_vals, _ = FAP(imagen)
  trk_vals, _ = T_rk(imagen)

  imagen_nueva = np.zeros(imagen.shape)
  alto, ancho = imagen.shape

  for j in range(alto):
    for i in range(ancho):
      # Usamos el valor del pixel como índice para T_rk
      imagen_nueva[j][i] = trk_vals[imagen[j][i]]
  return imagen_nueva

ima_ecua = ecualizacion(imagen_gris)
histo_ecua = histo_casero(ima_ecua)


# List of image variables and their corresponding histograms
image_vars = {
    "Original (Gris)": (imagen_gris, histo_imagen),
    "Extendida": (imagen_extendida, histo_extendido),
    "Escala": (ima_escala, histo_escala),
    "Ecualizada": (ima_ecua, histo_ecua)
}

print(f"{'Variable':<20} | {'Shape':<15} | {'Dtype':<15} | {'Max Value':<10}")
print("-" * 70)

for name, (img, histo) in image_vars.items():
    # Check image properties
    img_shape = img.shape
    img_dtype = img.dtype
    img_max = np.max(img)

    # Check histogram
    histo_len = len(histo)

    print(f"{name:<20} | {str(img_shape):<15} | {str(img_dtype):<15} | {img_max:.2f}")

    # Ensure histograms have 256 bins
    assert histo_len == 256, f"Error: {name} histogram has {histo_len} bins instead of 256."

print("\nAll variables verified and ready for grid visualization.")

"""## Crear cuadrícula de comparación

### Subtask:
Generate a 2x4 comparison grid to visualize the four image processing methods and their respective histograms.

"""

import matplotlib.pyplot as plt

# Datos a iterar, creamos un arreglo con los datos a graficar
methods_data = [
    ("Original", imagen_gris, histo_imagen),
    ("Extension", imagen_extendida, histo_extendido),
    ("Parametric Scaling", ima_escala, histo_escala),
    ("Equalization", ima_ecua, histo_ecua)
]

# Creamos el lienzo general
fig, axes = plt.subplots(2, 4, figsize=(20, 10))


# Iterar
for i, (name, img, histo) in enumerate(methods_data):
    # 3. Populate top row with images
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f"{name} Image")
    axes[0, i].axis('off')

    # Using bar for better histogram visualization, matching the 0-255 range
    axes[1, i].bar(range(256), histo, color='gray', width=1.0)
    axes[1, i].set_title(f"{name} Histogram")
    axes[1, i].set_xlim([0, 255])
    axes[1, i].grid(True, alpha=0.3)

# 6. Final layout adjustment
plt.tight_layout()
plt.show()