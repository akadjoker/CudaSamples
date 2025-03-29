import os
import cv2
import albumentations as A
import numpy as np

# Diretórios
IMG_DIR = "frames_portagem/images"
LABEL_DIR = "frames_portagem/labels"
OUT_IMG_DIR = "frames_portagem_augmented/images"
OUT_LABEL_DIR = "frames_portagem_augmented/labels"
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# Transforms
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))

image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))])
counter = 0

for img_file in image_files:
    img_path = os.path.join(IMG_DIR, img_file)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_file)

    if not os.path.exists(label_path):
        print(f"⚠️ Sem label para {img_file}, ignorando.")
        continue

    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    # Ler labels
    bboxes = []
    class_labels = []
    with open(label_path, "r") as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            bboxes.append([x, y, w, h])
            class_labels.append(int(cls))

    for i in range(5):
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]

            # Clamp boxes para estarem dentro dos limites
            clamped_bboxes = []
            for box in aug_bboxes:
                x, y, w, h = box
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)
                clamped_bboxes.append([x, y, w, h])

            # Guardar imagem
            out_img = f"aug_{counter:05}.jpg"
            cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img), aug_img)

            # Guardar labels
            out_label = f"aug_{counter:05}.txt"
            with open(os.path.join(OUT_LABEL_DIR, out_label), "w") as f:
                for box, cls in zip(clamped_bboxes, aug_labels):
                    f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")

            counter += 1

        except Exception as e:
            print(f"❌ Erro ao processar {img_file} (aug {i}): {e}")

print(f"\n✅ Aumento completo! {counter} imagens geradas.")


