"""데이터셋 준비용 보조 유틸리티

1) match_images_to_labels: 라벨(.txt) 폴더를 기준으로, 원본 이미지 폴더에서
   대응하는 이미지 파일을 찾아 같은 폴더로 복사
2) remap_to_single_class: 이미지+라벨을 새 폴더로 복사하면서 모든 클래스 ID를
   0으로 통일 (예: 결함 검출을 단일 클래스 문제로 단순화할 때)
"""
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def match_images_to_labels(image_source_dir: str, label_dir: str) -> None:
    """label_dir 안의 .txt 파일 이름 기준으로 image_source_dir에서 이미지를 찾아 label_dir로 복사"""
    image_source_dir = Path(image_source_dir)
    label_dir = Path(label_dir)

    txt_stems = [f.stem for f in label_dir.glob("*.txt")]
    copied_count = 0
    not_found_count = 0

    for stem in txt_stems:
        found = False
        for ext in IMAGE_EXTENSIONS:
            source_image = image_source_dir / f"{stem}{ext}"
            if source_image.exists():
                shutil.copy2(source_image, label_dir / f"{stem}{ext}")
                copied_count += 1
                found = True
                print(f"복사 완료: {stem}{ext}")
                break
        if not found:
            print(f"이미지를 찾을 수 없음: {stem}")
            not_found_count += 1

    print(f"\n총 {copied_count}개 이미지 복사 완료 / {not_found_count}개 찾을 수 없음")


def remap_to_single_class(source_dir: str, dest_dir: str) -> None:
    """이미지 복사 + 라벨의 클래스 ID를 모두 0으로 변경해 dest_dir에 저장"""
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(exist_ok=True)

    for ext in IMAGE_EXTENSIONS[:4]:
        for img in source.glob(f"*{ext}"):
            shutil.copy2(img, dest / img.name)
    print("✅ 이미지 복사 완료")

    for txt in source.glob("*.txt"):
        new_lines = []
        with open(txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    new_lines.append("0 " + " ".join(parts[1:]) + "\n")
        with open(dest / txt.name, "w") as f:
            f.writelines(new_lines)
    print(f"✅ 라벨 수정 및 복사 완료\n결과: {dest}")
