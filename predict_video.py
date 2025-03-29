import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from treino import PilotNet
from PIL import Image

MODEL_PATH = 'steering_model_best.pth'
#VIDEO_PATH = '/home/djoker/code/Seame/sea:me/session_20250324_170653/video_20250324_172157_whatsapp.mp4'
VIDEO_PATH = '/home/djoker/code/Seame/sea:me/session_20250324_170653/video_20250324_165443_.mp4'

IMG_SIZE = (66, 200)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PilotNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Não foi possível abrir o vídeo.")
    exit()

print("🔁 A correr vídeo em loop. Pressiona 'q' para sair.")

while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_angle = model(input_tensor).item()

    cv2.putText(frame, f"Steering: {predicted_angle:.3f}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Modelo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

