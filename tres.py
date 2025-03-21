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

def label_objects(image):
    height, width = image.shape
    labeled = np.zeros_like(image, dtype=int)
    label = 1
    min_area = 500  # Limite mínimo para considerar um objeto válido

    def flood(i, j, label):
        stack = [(i, j)]
        pixels = []
        while stack:
            x, y = stack.pop()
            if labeled[x, y] == 0 and image[x, y] == 1:
                labeled[x, y] = label
                pixels.append((x, y))
                if x > 0: stack.append((x-1, y))
                if x < height-1: stack.append((x+1, y))
                if y > 0: stack.append((x, y-1))
                if y < width-1: stack.append((x, y+1))
        return pixels

    object_sizes = []
    
    for i in range(height):
        for j in range(width):
            if image[i, j] == 1 and labeled[i, j] == 0:
                pixels = flood(i, j, label)
                if len(pixels) >= min_area:  # Filtra objetos muito pequenos
                    object_sizes.append(len(pixels))
                    label += 1
                else:
                    for x, y in pixels:
                        labeled[x, y] = 0  # Remove ruídos menores que o limite
    
    return labeled, len(object_sizes)

def show_results(original, binarized, eroded, dilated, labeled, count):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    titles = ["Original", "Binarizada", "Erosão", "Dilatação", f"Objetos: {count}"]

    images = [original, binarized * 255, eroded * 255, dilated * 255, labeled * (255 // max(count, 1))]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.show()

# Carregar imagem
image_path = "training/2.jpg"
gray_image = load_image(image_path)

# Processamento
binarized = binarize(gray_image)
eroded = erode(binarized)
dilated = dilate(eroded, kernel_size= 3)  # Dilatação mais forte para unir objetos quebrados
labeled, obj_count = label_objects(dilated)

# Exibir resultados
show_results(gray_image, binarized, eroded, dilated, labeled, obj_count)

#  dilatação 5 a imagem 7 fica certa, imagem 5 fica ruim
#  dilatação 3 a imagem 5 fica certa, imagem 7 fica ruim