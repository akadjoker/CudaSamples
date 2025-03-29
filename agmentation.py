import os
import cv2
import numpy as np
from pathlib import Path
import random


def carregar_imagem(caminho_imagem):
    if not os.path.exists(caminho_imagem):
        print(f"Erro: A imagem {caminho_imagem} não existe.")
        return None
    return cv2.imread(caminho_imagem)


def flip_horizontal(imagem):
    return cv2.flip(imagem, 1)

def rotacao(imagem, angulo_range=10):
    altura, largura = imagem.shape[:2]
    angulo = random.uniform(-angulo_range, angulo_range)
    centro = (largura // 2, altura // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(imagem, matriz, (largura, altura))

def shear(imagem, shear_range=0.2):
    altura, largura = imagem.shape[:2]
    shear_factor = random.uniform(-shear_range, shear_range)
    matriz_shear = np.array([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(imagem, matriz_shear, (largura, altura))

def skew(imagem):
    altura, largura = imagem.shape[:2]
    desloca = 30
    pts1 = np.float32([[0, 0], [largura, 0], [0, altura], [largura, altura]])
    pts2 = np.float32([
        [random.randint(0, desloca), random.randint(0, desloca)],
        [largura - random.randint(0, desloca), random.randint(0, desloca)],
        [random.randint(0, desloca), altura - random.randint(0, desloca)],
        [largura - random.randint(0, desloca), altura - random.randint(0, desloca)],
    ])
    matriz = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(imagem, matriz, (largura, altura))

def ruido(imagem, intensidade=0.5):
    ruido = np.random.normal(0, intensidade, imagem.shape).astype(np.uint8)
    return cv2.add(imagem, ruido)

def brilho_contraste(imagem, brilho=30, contraste=1.2):
    brilho_rand = random.randint(-brilho, brilho)
    contraste_rand = random.uniform(1.0, contraste)
    return cv2.convertScaleAbs(imagem, alpha=contraste_rand, beta=brilho_rand)

def hsv_jitter(imagem, s_jitter=0.3, v_jitter=0.3):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(1 - s_jitter, 1 + s_jitter)
    hsv[:, :, 2] *= random.uniform(1 - v_jitter, 1 + v_jitter)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def aplicar_aumentacoes(imagem_base, caminho_saida, nome_base):
    Path(caminho_saida).mkdir(parents=True, exist_ok=True)

    aumentacoes = {
        "Flip": flip_horizontal,
        "Rotate": rotacao,
        "Shear": shear,
        "Skew": skew,
        "Noise": ruido,
        "BrightnessContrast": brilho_contraste,
        "HSV": hsv_jitter
    }

    for nome, funcao in aumentacoes.items():
        try:
            imagem_aug = funcao(imagem_base.copy())
            imagem_aug = cv2.resize(imagem_aug, (640, 640))
            caminho = os.path.join(caminho_saida, f"{nome_base}_{nome}.jpg")
            cv2.imwrite(caminho, imagem_aug)
            print(f"Aumentada: {caminho}")
        except Exception as e:
            print(f"Erro em {nome}: {e}")


def processar_pasta(pasta_entrada, pasta_saida):
    extensoes = ['.jpg', '.jpeg', '.png']
    for raiz, _, ficheiros in os.walk(pasta_entrada):
        for nome in ficheiros:
            if Path(nome).suffix.lower() in extensoes:
                caminho = os.path.join(raiz, nome)
                imagem = carregar_imagem(caminho)
                if imagem is None:
                    continue
                try:
                    imagem_base = cv2.resize(imagem, (640, 640))
                    nome_base = Path(nome).stem
                    destino_base = os.path.join(pasta_saida, f"{nome_base}_resized.jpg")
                    cv2.imwrite(destino_base, imagem_base)
                    print(f"Original redimensionada: {destino_base}")
                    aplicar_aumentacoes(imagem_base, pasta_saida, nome_base)
                except Exception as e:
                    print(f"Erro a processar {nome}: {e}")


# Exemplo de uso
if __name__ == "__main__":
    pasta_input = "/home/djoker/code/cuda/portagens/imagens"
    pasta_output = "/home/djoker/code/cuda/portagens/new"
    processar_pasta(pasta_input, pasta_output)

