"""OR 이미지 -> OR & TR 이미지 셋 생성 (포스터 [그림 3])

원본(OR) n개에 회전(_r)/밝기(_b)/노이즈(_g) 3종을 각각 적용해 총 4n개의
OR & TR 이미지 셋을 만든다. YOLO 라벨(.txt)이 있으면 함께 변환/복사한다.

사용 예:
    python -m scripts.run_traditional_aug \
        --source /path/to/gan_train \
        --dest /path/to/gan_train_tr
"""
import argparse
import shutil
from pathlib import Path

import cv2

from src.traditional_aug.transforms import (
    rotate_90_left,
    rotate_bbox_90_left,
    increase_brightness,
    add_gaussian_noise,
)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def augment_dataset(source_folder: str, dest_folder: str) -> None:
    source = Path(source_folder)
    dest = Path(dest_folder)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"📁 {source_folder} → {dest_folder}")
    print("=" * 70)

    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(source.glob(f"*{ext}"))

    total_files = len(image_files)
    processed = 0

    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ 이미지 로드 실패: {img_path.name}")
                continue

            label_path = img_path.with_suffix(".txt")
            stem = img_path.stem
            ext = img_path.suffix

            # 1. 원본 복사
            shutil.copy2(img_path, dest / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, dest / label_path.name)

            # 2. 90도 회전 (_r)
            img_rotated = rotate_90_left(img)
            cv2.imwrite(str(dest / f"{stem}_r{ext}"), img_rotated)
            if label_path.exists():
                with open(label_path, "r") as f:
                    lines = f.readlines()
                rotated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = parts[0]
                        x_c, y_c, w, h = map(float, parts[1:5])
                        new_x, new_y, new_w, new_h = rotate_bbox_90_left(x_c, y_c, w, h)
                        rotated_lines.append(
                            f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n"
                        )
                with open(dest / f"{stem}_r.txt", "w") as f:
                    f.writelines(rotated_lines)

            # 3. 밝기 증가 (_b) - 라벨은 그대로
            img_bright = increase_brightness(img, value=30)
            cv2.imwrite(str(dest / f"{stem}_b{ext}"), img_bright)
            if label_path.exists():
                shutil.copy2(label_path, dest / f"{stem}_b.txt")

            # 4. 가우시안 노이즈 (_g) - 라벨은 그대로
            img_noisy = add_gaussian_noise(img, mean=0, sigma=10)
            cv2.imwrite(str(dest / f"{stem}_g{ext}"), img_noisy)
            if label_path.exists():
                shutil.copy2(label_path, dest / f"{stem}_g.txt")

            processed += 1
            if processed % 10 == 0:
                print(f"  처리 중... {processed}/{total_files}")

        except Exception as e:
            print(f"❌ 오류 발생 ({img_path.name}): {e}")
            continue

    final_images = len(list(dest.glob("*.jpg"))) + len(list(dest.glob("*.png")))
    final_labels = len(list(dest.glob("*.txt")))
    print(f"\n✅ 완료! 원본 {total_files}개 -> 총 {total_files * 4}개")
    print(f"📊 최종: 이미지 {final_images}개, 라벨 {final_labels}개")


def main():
    parser = argparse.ArgumentParser(description="OR -> OR & TR 이미지 셋 생성")
    parser.add_argument("--source", required=True, help="원본(OR) 이미지+라벨 폴더")
    parser.add_argument("--dest", required=True, help="OR & TR 이미지 셋 저장 폴더")
    args = parser.parse_args()
    augment_dataset(args.source, args.dest)


if __name__ == "__main__":
    main()
