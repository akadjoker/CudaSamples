import os
import glob
import random
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import PilotNet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# ============================
# Configurações
# ============================
DATASET_DIR = 'dataset'
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
IMG_SIZE = (66, 200)  # Altura, Largura (como no PilotNet)

# ============================
# Função para balancear o dataset
# ============================
def balance_dataset(image_paths, angles, threshold=0.05, keep_ratio=0.1):
    new_paths = []
    new_angles = []
    for img, angle in zip(image_paths, angles):
        if abs(angle) < threshold:
            if random.random() < keep_ratio:
                new_paths.append(img)
                new_angles.append(angle)
        else:
            new_paths.append(img)
            new_angles.append(angle)
    return np.array(new_paths), np.array(new_angles)

# ============================
# Dataset PyTorch com augmentation
# ============================
class SteeringDataset(Dataset):
    def __init__(self, image_paths, steering_angles, augment=False):
        self.image_paths = image_paths
        self.steering_angles = steering_angles
        self.augment = augment

        self.transform_basic = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])

        self.transform_augmented = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        angle = self.steering_angles[idx]

        if self.augment and random.random() < 0.5:
            #Simage = transforms.functional.hflip(image)
            #angle *= -1
            image = self.transform_augmented(image)
        else:
            image = self.transform_basic(image)

        return image, torch.tensor(angle, dtype=torch.float32)



# ============================
# Carregar dados do CSV
# ============================
def load_sessions_data():
    all_image_paths = []
    all_steering_angles = []
    session_dirs = glob.glob(os.path.join(DATASET_DIR, 'session_*'))

    if not session_dirs:
        raise ValueError(f"Nenhuma pasta de sessão encontrada em {DATASET_DIR}")

    for session_dir in session_dirs:
        csv_path = os.path.join(session_dir, 'steering_data.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if 'image_path' not in df.columns or 'steering' not in df.columns:
            continue

        for _, row in df.iterrows():
            image_path = os.path.join(session_dir, row['image_path'])
            if os.path.exists(image_path):
                all_image_paths.append(image_path)
                all_steering_angles.append(float(row['steering']))

    return np.array(all_image_paths), np.array(all_steering_angles)

# ============================
# Treinamento
# ============================
if __name__ == '__main__':
    device = torch.device('cuda')

    image_paths, angles = load_sessions_data()
    #image_paths, angles = balance_dataset(image_paths, angles, threshold=0.05, keep_ratio=0.15)

    dataset = SteeringDataset(image_paths, angles, augment=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)


    model = PilotNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                loss = loss_fn(preds, targets)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Época {epoch+1}/{EPOCHS} - Treino Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), 'steering_model.pth')
    print("\n✅ Modelo salvo em 'steering_model.pth'")

    # ============================
    # Gerar gráfico
    # ============================
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Treino')
    plt.plot(val_losses, label='Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (MSE)')
    plt.title('Evolução do Treinamento')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_plot.png')
    plt.show()
