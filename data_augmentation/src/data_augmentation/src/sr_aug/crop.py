"""SR(2배 업스케일) 이미지를 결함 위치 중심으로 크롭해 원본 크기로 복원 (포스터 [그림 2])"""
from pathlib import Path

import cv2


def center_crop_images(input_dir: str, output_dir: str, target_size: int = 640) -> None:
    """이미지를 중심 기준으로 target_size x target_size 정사각형으로 크롭"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.jpg"))
    processed_count = 0
    error_count = 0

    for image_file in image_files:
        try:
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"⚠️  이미지 읽기 실패: {image_file.name}")
                error_count += 1
                continue

            h, w = img.shape[:2]
            center_x, center_y = w // 2, h // 2
            half_size = target_size // 2
            crop_x1, crop_y1 = center_x - half_size, center_y - half_size
            crop_x2, crop_y2 = center_x + half_size, center_y + half_size

            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if cropped.shape[0] != target_size or cropped.shape[1] != target_size:
                print(f"⚠️  크롭 크기 오류: {image_file.name} - {cropped.shape}")
                error_count += 1
                continue

            cv2.imwrite(str(output_dir / image_file.name), cropped)
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"처리 중... {processed_count}개 완료")

        except Exception as e:
            print(f"❌ 오류 발생: {image_file.name} - {e}")
            error_count += 1

    print(f"\n✅ 처리 완료: {processed_count}개 / ❌ 오류: {error_count}개")
    print(f"📁 저장 위치: {output_dir}")
