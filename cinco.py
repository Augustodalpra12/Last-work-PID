import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    img = Image.open(path).convert('L')  # Converte para escala de cinza
    return np.array(img)

def apply_box_filter(image, kernel_size):
    height, width = image.shape
    pad = kernel_size // 2  # Para manter o tamanho original, evitando problemas a calcular os pixeis das bordas
    padded_image = np.pad(image, pad, mode='edge')  # Preenchimento das bordas
    
    output = np.zeros_like(image, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.mean(region).astype(np.uint8)
    
    return output

def show_images(original, filtered_images, kernel_sizes):
    fig, axes = plt.subplots(1, len(filtered_images) + 1, figsize=(15, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    for ax, img, k in zip(axes[1:], filtered_images, kernel_sizes):
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Filtro {k}x{k}")
        ax.axis('off')
    
    plt.show()

image_path = "img/teste.webp" 

# Carregar imagem
gray_image = load_image(image_path)

# Aplicar filtros
kernel_sizes = [2, 3, 5, 7]
filtered_images = [apply_box_filter(gray_image, k) for k in kernel_sizes]

# Mostrar imagens
show_images(gray_image, filtered_images, kernel_sizes)
