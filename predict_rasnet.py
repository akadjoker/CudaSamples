import cv2
import torch
import numpy as np
from torchvision import transforms
from model_resnet import ResNetSteering
from PIL import Image
import math

# ============================
# Configurações
# ============================
MODEL_PATH = 'steering_model_resnet.pth'
VIDEO_PATH = '/home/djoker/code/Seame/sea:me/session_20250324_170653/video_final.mp4'
#VIDEO_PATH = '/home/djoker/code/Seame/sea:me/session_20250324_170653/video_20250324_165443_.mp4'
IMG_SIZE = (224, 224)

# ============================
# Transform para ResNet
# ============================
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================
# Carregar modelo
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetSteering(freeze_features=False).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ============================
# Leitura do vídeo
# ============================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
        continue

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        steering = model(tensor).item()

    # ============================
    # Desenhar indicador gráfico de direção
    # ============================
    steering_angle = max(-1.0, min(1.0, steering))  # limitar
    angle_rad = steering_angle * math.radians(45)  # escala para [-45º, 45º]

    cx, cy = 100, 100
    r = 40
    x = int(cx + r * math.sin(angle_rad))
    y = int(cy - r * math.cos(angle_rad))

    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
    cv2.line(frame, (cx, cy), (x, y), (0, 255, 0), 3)

    # Mostrar
    cv2.imshow('Steering Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
