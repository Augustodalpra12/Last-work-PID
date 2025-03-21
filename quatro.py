from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Direções da Cadeia de Freeman (sentido horário)
freeman_dirs = {
    (0, 1): 0,   # Direita
    (-1, 1): 1,  # Superior direita
    (-1, 0): 2,  # Cima
    (-1, -1): 3, # Superior esquerda
    (0, -1): 4,  # Esquerda
    (1, -1): 5,  # Inferior esquerda
    (1, 0): 6,   # Baixo
    (1, 1): 7    # Inferior direita
}

# (anti-horário) para percorrer os contornos
movimentos = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]

def binarizar_imagem(img, limiar=128):
    img_gray = img.convert("L")  
    img_array = np.array(img_gray)
    binaria = (img_array < limiar).astype(np.uint8)  # 0 = fundo, 1 = objeto
    return binaria

def encontrar_ponto_inicial(img_bin): # achou 1, achou o objeto
    for y in range(img_bin.shape[0]):
        for x in range(img_bin.shape[1]):
            if img_bin[y, x] == 1:
                return y, x
    return None

def encontrar_contorno(img_bin, ponto_inicial):
    y, x = ponto_inicial
    contorno = [(y, x)]
    direcao_atual = 0  # Começamos buscando para a direita
    
    for i in range(8): # usando as 8 direções, tenta encontrar o vizinho valido
        dy, dx = movimentos[i]
        ny, nx = y + dy, x + dx
        if 0 <= ny < img_bin.shape[0] and 0 <= nx < img_bin.shape[1] and img_bin[ny, nx] == 1:
            y, x = ny, nx
            contorno.append((y, x))
            direcao_atual = i
            break
    
    while True:
        encontrou = False
        for i in range(8):
            direcao = (direcao_atual + i) % 8 
            dy, dx = movimentos[direcao]
            ny, nx = y + dy, x + dx

            if 0 <= ny < img_bin.shape[0] and 0 <= nx < img_bin.shape[1] and img_bin[ny, nx] == 1:
                if (ny, nx) == ponto_inicial and len(contorno) > 1:
                    return contorno
                if (ny, nx) not in contorno:
                    contorno.append((ny, nx))
                    y, x = ny, nx
                    direcao_atual = direcao
                    encontrou = True
                    #print(direcao_atual)
                    break
        
        if not encontrou:
            break  # Para se não encontrar mais pixels de borda
    
    return contorno

def gerar_cadeia_freeman(contorno): # percorre os pontos, e calcula a diferença entre dois pontos consecutivos
    cadeia = []
    for i in range(1, len(contorno)):
        dy, dx = contorno[i][0] - contorno[i - 1][0], contorno[i][1] - contorno[i - 1][1]
        #print("Primeiro contorno", contorno[i][0], "Segundo contorno", contorno[i-1][0])
        if (dy, dx) in freeman_dirs:
            cadeia.append(freeman_dirs[(dy, dx)])
            #print(freeman_dirs[(dy, dx)])
    return cadeia

# Carregar a imagem e processar
# triangulo
#img = Image.open("img/triangulo_amarelo.jpg") 
# corinthians
#img = Image.open("img/corinthians.jpg") 
# quadrado
# img = Image.open("img/quadrado.png")

# quadrado paint
# img = Image.open("img/quadrado_paint.png")

# quadrado fino
img = Image.open("img/quadrado_fino.png")


binaria = binarizar_imagem(img)

# Encontrar contorno e calcular a cadeia de Freeman
ponto_inicial = encontrar_ponto_inicial(binaria)
if ponto_inicial:
    contorno = encontrar_contorno(binaria, ponto_inicial)
    if len(contorno) > 1:
        cadeia_freeman = gerar_cadeia_freeman(contorno)
        print("Cadeia de Freeman:", cadeia_freeman)
    else:
        print("Contorno não encontrado corretamente.")
else:
    print("Nenhum objeto encontrado na imagem.")


# Plotar a imagem binarizada
plt.imshow(binaria, cmap='gray')
plt.title('Imagem Binarizada')
plt.axis('off')
plt.show()