import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

#======================================================================
#                     IMAGEM
#======================================================================

# Abrir a imagem e converter para escala de cinza usando skimage
img = Image.open('img/triangulo_amarelo.jpg')
img_gray = color.rgb2gray(np.array(img))

# Normalizar para valores entre 0 e 255
img_gray = (img_gray * 255).astype(np.uint8)

#======================================================================
#                     IMAGEM
#======================================================================

#======================================================================
#                    WaterShed
#======================================================================


#  marcadores watershed
def get_markers(image):
    markers = np.zeros_like(image, dtype=np.int32)
    markers[image < np.percentile(image, 50)] = 1  # Região de fundo
    markers[image > np.percentile(image, 50)] = 2  # Região de objeto
    return markers


#WaterShed
def watershed(image, markers):
    rows, cols = image.shape
    segmented = np.zeros_like(image, dtype=np.int32)
    
    # Inicializando as filas para cada marcador
    queue = []
    for i in range(rows):
        for j in range(cols):
            if markers[i, j] > 0:
                queue.append((i, j, markers[i, j]))
    
    while queue:
        i, j, label = queue.pop(0)
        if segmented[i, j] == 0:  # Se ainda não foi visitado
            segmented[i, j] = label
            
            # Vizinhos 4-conectados
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and segmented[ni, nj] == 0:
                    queue.append((ni, nj, label))
    
    return segmented

# Gerando os marcadores
markers = get_markers(img_gray)

# Aplicando Watershed
segmented_image = watershed(img_gray, markers)

#======================================================================
#                    WaterShed
#======================================================================

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_gray, cmap='gray')
ax[0].set_title("Imagem Original")
ax[0].axis("off")

ax[1].imshow(segmented_image, cmap='jet')
ax[1].set_title("Segmentação Watershed")
ax[1].axis("off")



plt.show()