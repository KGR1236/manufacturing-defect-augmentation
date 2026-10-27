"""x2 업스케일된 SR 이미지에 맞춰 YOLO 라벨의 width/height를 2배로 변환

주의: 크롭 이후에는 라벨 좌표 자체는 원본과 동일한 상대 위치를 유지하지만,
업스케일 배율만큼 커진 박스 크기를 반영해야 하므로 width/height만 2배로 조정한다.
"""
from pathlib import Path


def transform_yolo_labels(image_dir: str, label_dir: str, output_dir: str) -> None:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.jpg"))
    processed_count = 0
    skipped_count = 0

    for image_file in image_files:
        label_file = label_dir / f"{image_file.stem}.txt"
        if not label_file.exists():
            print(f"⚠️  라벨 파일 없음: {label_file.name}")
            skipped_count += 1
            continue

        output_file = output_dir / f"{image_file.stem}.txt"
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"⚠️  잘못된 형식: {label_file.name} - {line}")
                continue

            class_id = parts[0]
            x_center, y_center = float(parts[1]), float(parts[2])
            width, height = float(parts[3]) * 2, float(parts[4]) * 2
            width, height = min(width, 1.0), min(height, 1.0)

            x1 = max(0, x_center - width / 2)
            y1 = max(0, y_center - height / 2)
            x2 = min(1, x_center + width / 2)
            y2 = min(1, y_center + height / 2)

            adj_w, adj_h = x2 - x1, y2 - y1
            adj_x, adj_y = (x1 + x2) / 2, (y1 + y2) / 2

            new_lines.append(f"{class_id} {adj_x:.6f} {adj_y:.6f} {adj_w:.6f} {adj_h:.6f}")

        with open(output_file, "w") as f:
            f.write("\n".join(new_lines))
        processed_count += 1

    print(f"\n✅ 처리 완료: {processed_count}개 / ⚠️  건너뜀: {skipped_count}개")
    print(f"📁 저장 위치: {output_dir}")
