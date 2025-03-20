import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    img = Image.open(path).convert('L')  # Converte para escala de cinza
    return np.array(img)

def binarize(image):
    threshold = np.mean(image) - 10  # Ajusta o limiar automaticamente
    return (image < threshold).astype(np.uint8)  # Inverte para preto=1 e branco=0

def erode(image, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size] 
            output[i, j] = np.min(region)  # Mantém apenas áreas completamente preenchidas
            
    return output

def dilate(image, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.max(region)  # Expande objetos conectados
            
    return output

def distance_transform(binary):
    # Usaremos uma transformada de distância aproximada (distância Manhattan)
    height, width = binary.shape
    dt = np.where(binary==1, 9999, 0).astype(np.float32)
    
    # Passagem para frente
    for i in range(height):
        for j in range(width):
            if binary[i, j] == 1:
                min_val = 9999
                if i > 0:
                    min_val = min(min_val, dt[i-1, j])
                if j > 0:
                    min_val = min(min_val, dt[i, j-1])
                dt[i, j] = min(dt[i, j], min_val + 1)
    
    # Passagem para trás
    for i in range(height-1, -1, -1):
        for j in range(width-1, -1, -1):
            if binary[i, j] == 1:
                min_val = 9999
                if i < height - 1:
                    min_val = min(min_val, dt[i+1, j])
                if j < width - 1:
                    min_val = min(min_val, dt[i, j+1])
                dt[i, j] = min(dt[i, j], min_val + 1)
                
    return dt

def find_markers(dt):
    # Detecta marcadores como máximos locais (vizinhança 4)
    height, width = dt.shape
    markers = np.zeros_like(dt, dtype=np.int32)
    label = 1
    for i in range(1, height-1):
        for j in range(1, width-1):
            center = dt[i, j]
            if center > dt[i-1, j] and center > dt[i+1, j] and center > dt[i, j-1] and center > dt[i, j+1]:
                markers[i, j] = label
                label += 1
    return markers, label - 1

def watershed(dt, markers):
    height, width = dt.shape
    labels = np.copy(markers)
    
    # Inicializa uma lista com os pixels vizinhos dos marcadores
    queue = []
    for i in range(height):
        for j in range(width):
            if markers[i, j] > 0:
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < height and 0 <= nj < width:
                        if labels[ni, nj] == 0 and dt[ni, nj] > 0:
                            queue.append((ni, nj))
                            
    # Ordena a fila com base no valor da distância (processa os de maior valor primeiro)
    queue = sorted(queue, key=lambda p: dt[p[0], p[1]], reverse=True)
    
    while queue:
        i, j = queue.pop(0)
        neighbor_labels = set()
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < height and 0 <= nj < width:
                if labels[ni, nj] > 0:
                    neighbor_labels.add(labels[ni, nj])
        if len(neighbor_labels) == 1:
            labels[i, j] = neighbor_labels.pop()
        elif len(neighbor_labels) > 1:
            labels[i, j] = -1  # Fronteira entre regiões
        
        # Adiciona vizinhos não processados
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < height and 0 <= nj < width:
                if labels[ni, nj] == 0 and dt[ni, nj] > 0 and (ni, nj) not in queue:
                    queue.append((ni, nj))
        # Reordena a fila
        queue = sorted(queue, key=lambda p: dt[p[0], p[1]], reverse=True)
    
    return labels

def label_objects_watershed(binary):
    # Calcula a transformada de distância
    dt = distance_transform(binary)
    # Encontra marcadores (máximos locais da DT)
    markers, num_markers = find_markers(dt)
    # Aplica o watershed
    labels = watershed(dt, markers)
    # Remove fronteiras marcadas como -1
    labels[labels == -1] = 0
    
    # Filtra objetos pequenos
    min_area = 500
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        if lab == 0:
            continue
        if np.sum(labels == lab) < min_area:
            labels[labels == lab] = 0
    
    # Reorganiza os rótulos para que fiquem sequenciais
    new_labels = np.zeros_like(labels, dtype=np.int32)
    current_label = 1
    for lab in np.unique(labels):
        if lab == 0:
            continue
        new_labels[labels == lab] = current_label
        current_label += 1
        
    return new_labels, current_label - 1

def show_results(original, binarized, eroded, dilated, labeled, count):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    titles = ["Original", "Binarizada", "Erosão", "Dilatação", f"Objetos: {count}"]

    # Para visualização, escalamos os valores dos rótulos
    labeled_vis = (labeled * (255 // max(count, 1))).astype(np.uint8)
    images = [original, binarized * 255, eroded * 255, dilated * 255, labeled_vis]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# Carregar imagem
image_path = "training/5.jpg"
gray_image = load_image(image_path)

# Processamento
binarized = binarize(gray_image)
eroded = erode(binarized)
dilated = dilate(eroded, kernel_size=3)  # Dilatação mais forte para unir objetos quebrados

# Segmentação com watershed
labeled, obj_count = label_objects_watershed(dilated)

# Exibir resultados
show_results(gray_image, binarized, eroded, dilated, labeled, obj_count)
