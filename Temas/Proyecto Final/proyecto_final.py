import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb_to_gray(img):
    """Convierte una imagen RGB a escala de grises mediante suma ponderada."""
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

def convolve2d_fast(image, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Padding
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(image)
    
    # Desplazamiento y suma multiplicativa (con slicing para eficiencia)
    for i in range(k_h):
        for j in range(k_w):
            out += padded[i:i+image.shape[0], j:j+image.shape[1]] * kernel[i, j]
    return out

def harris_corner_detector(img_gray, k=0.04, window_size=3):
    # Filtros de Sobel para gradientes espaciales
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1], 
                        [ 0,  0,  0], 
                        [ 1,  2,  1]])
    
    Ix = convolve2d_fast(img_gray, sobel_x)
    Iy = convolve2d_fast(img_gray, sobel_y)
    
    # Productos de las derivadas
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
    # Suavizado (Kernel Gaussiano simple de 3x3)
    gaussian_kernel = np.array([[1, 2, 1], 
                                [2, 4, 2], 
                                [1, 2, 1]]) / 16.0
    
    Sxx = convolve2d_fast(Ixx, gaussian_kernel)
    Syy = convolve2d_fast(Iyy, gaussian_kernel)
    Sxy = convolve2d_fast(Ixy, gaussian_kernel)
    
    # Respuesta de Harris
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)
    
    return R

def non_maximum_suppression(R, threshold_ratio=0.01, min_distance=10):
    """Aplica supresión de no máximos para aislar las esquinas detectadas"""
    threshold = np.max(R) * threshold_ratio
    # Encontrar coordenadas que superen el umbral
    coords = np.argwhere(R > threshold) # Encuentra las posiciones donde se cumple la condicion 
    
    # Ordenar por el valor de respuesta de Harris (de mayor a menor)
    responses = R[coords[:, 0], coords[:, 1]]
    coords = coords[np.argsort(responses)[::-1]]
    
    keep = []
    for r, c in coords:
        if not keep:
            keep.append((r, c))
            continue
            
        # Distancia de Chebyshev a los puntos ya guardados
        keep_arr = np.array(keep)
        dist = np.max(np.abs(keep_arr - [r, c]), axis=1) 
        
        # Si está lo suficientemente lejos de los puntos más fuertes, se guarda
        if np.all(dist > min_distance):
            keep.append((r, c))
            
    return np.array(keep)

def calculate_distances(points):
    distances = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            dist = np.sqrt(np.sum((p1 - p2)**2))
            distances.append((p1, p2, dist))
    return distances

def main():
    # Lectura de imagen
    ruta = r'Temas/Proyecto Final/Cuadros.jpeg'
    img_bgr = cv2.imread(ruta) 
    if img_bgr is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
        return
        
    img_rgb = img_bgr[:, :, ::-1] # BGR a RGB
    img_gray = rgb_to_gray(img_rgb)
    
    # Detección de características
    R = harris_corner_detector(img_gray, k=0.04)
    
    # Extracción de puntos: 
    corners = non_maximum_suppression(R, threshold_ratio=0.05, min_distance=25)
    
    print(f"Se encontraron {len(corners)} esquinas.")
    
    # Graficas
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    
    if len(corners) > 0:
        # Nota: corners tiene formato (fila, columna) -> (y, x)
        # Matplotlib scatter usa (x, y)
        x_coords = corners[:, 1]
        y_coords = corners[:, 0]
        plt.scatter(x_coords, y_coords, c='red', s=40, marker='x')
        
        # Línea entre los primeros dos puntos encontrados (por el algoritmo)
        if len(corners) >= 2:
            p1, p2 = corners[0], corners[1]
            dist = np.sqrt(np.sum((p1 - p2)**2))
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'c--', linewidth=2)
            print(f"Distancia de ejemplo entre dos puntos contiguos: {dist:.2f} píxeles")

    # plt.title('Detección de puntos de calibración (Harris)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()