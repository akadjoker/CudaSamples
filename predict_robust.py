import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from model_robust import RobustSteeringNet
from PIL import Image
import math
VIDEO_PATH = '/home/djoker/code/Seame/sea:me/session_20250324_170653/video_20250324_172656_.mp4'
MODEL_PATH = 'steering_model_robust.pth'
IMG_SIZE = (64, 64)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RobustSteeringNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Erro ao abrir vídeo.")
    exit()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v', 'XVID'
out = cv2.VideoWriter("model.mp4", fourcc, 30, (640, 480))    

def desenhar_angulo(frame, angle):
    h, w, _ = frame.shape
    centro = (w // 2, h - 30)
    comprimento = 100
    rad = angle * np.pi / 2  # valor normalizado entre -1 e 1
    x = int(centro[0] + comprimento * np.sin(rad))
    y = int(centro[1] - comprimento * np.cos(rad))
    cv2.arrowedLine(frame, centro, (x, y), (0, 255, 0), 3, tipLength=0.3)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #continue
        break

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        angle = model(input_tensor).item()

    steering_angle = max(-1.0, min(1.0, angle))  # limitar
    angle_rad = steering_angle * math.radians(45)  # escala para [-45º, 45º]

    cx, cy = 100, 100
    r = 40
    x = int(cx + r * math.sin(angle_rad))
    y = int(cy - r * math.cos(angle_rad))

    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
    cv2.line(frame, (cx, cy), (x, y), (0, 255, 0), 3)

    frame = desenhar_angulo(frame, angle)
    texto = f"Angle: {angle:.2f}"
    cv2.putText(frame, texto, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    out.write(frame)
    cv2.imshow("RobustSteeringNet", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
