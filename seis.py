import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    img = Image.open(path).convert('L')  # Converte para escala de cinza
    return np.array(img)

def segment_image(image):
    
    segmented = np.zeros_like(image, dtype=np.uint8)
    
    # Definição dos intervalos conforme a tabela
    ranges = [(0, 50, 25), (51, 100, 75), (101, 150, 125), (151, 200, 175), (201, 255, 255)]
    
    for low, high, value in ranges:
        mask = (image >= low) & (image <= high)
        segmented[mask] = value
    
    return segmented

def show_images(original, segmented):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(segmented, cmap='gray')
    axes[1].set_title("Segmentada")
    axes[1].axis('off')
    
    plt.show()

# Caminho da imagem
image_path = "img/jamal_murray.png"  # Substitua pelo caminho correto

# Carregar imagem
gray_image = load_image(image_path)

# Aplicar segmentação
segmented_image = segment_image(gray_image)

# Mostrar imagens
show_images(gray_image, segmented_image)
