"""증강/변환된 이미지의 YOLO 라벨을 랜덤 샘플링해 바운딩 박스로 시각화 (QA용)"""
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def visualize_yolo_labels(image_dir: str, label_dir: str, num_samples: int = 3) -> None:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_files = list(image_dir.glob("*.jpg"))
    num_samples = min(num_samples, len(image_files))
    selected_files = random.sample(image_files, num_samples)

    fig, axes = plt.subplots(1, num_samples, figsize=(6 * num_samples, 6))
    if num_samples == 1:
        axes = [axes]

    for idx, image_file in enumerate(selected_files):
        img = cv2.imread(str(image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        label_file = label_dir / f"{image_file.stem}.txt"
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img, f"Class {int(class_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    )
        else:
            print(f"⚠️  라벨 파일 없음: {label_file.name}")

        axes[idx].imshow(img)
        axes[idx].set_title(f"{image_file.name}\n{w}×{h}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()
