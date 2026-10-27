"""OR & TR 이미지 셋 -> SR 이미지 셋 생성 (포스터 [그림 2] 전체 파이프라인)

순서: x2 업스케일 -> 결함 위치 중심 크롭(원본 크기로 복원) -> 라벨 크기 변환

사용 예:
    python -m scripts.run_sr_conversion \
        --input /path/to/or_tr_images \
        --upscaled /path/to/tmp_upscaled \
        --cropped /path/to/sr_images \
        --label-dir /path/to/or_tr_images \
        --label-out /path/to/sr_images \
        --model-path /path/to/RealESRGAN_x2plus.pth \
        --crop-size 640
"""
import argparse

from src.sr_aug.upscale import upscale_images
from src.sr_aug.crop import center_crop_images
from src.sr_aug.label_transform import transform_yolo_labels


def main():
    parser = argparse.ArgumentParser(description="OR & TR -> SR 이미지 셋 생성")
    parser.add_argument("--input", required=True, help="업스케일할 원본 이미지 폴더 (OR & TR 전체)")
    parser.add_argument("--upscaled", required=True, help="x2 업스케일 결과 임시 저장 폴더")
    parser.add_argument("--cropped", required=True, help="크롭 후 최종 SR 이미지 저장 폴더")
    parser.add_argument("--label-dir", required=True, help="원본 YOLO 라벨(.txt) 폴더")
    parser.add_argument("--label-out", required=True, help="변환된 라벨 저장 폴더")
    parser.add_argument("--model-path", required=True, help="RealESRGAN_x2plus.pth 경로")
    parser.add_argument("--crop-size", type=int, default=640, help="크롭 후 최종 이미지 크기")
    args = parser.parse_args()

    print("[1/3] SR 업스케일 (x2)")
    upscale_images(args.input, args.upscaled, args.model_path, suffix="_s")

    print("\n[2/3] 결함 위치 중심 크롭")
    center_crop_images(args.upscaled, args.cropped, target_size=args.crop_size)

    print("\n[3/3] YOLO 라벨 크기 변환")
    transform_yolo_labels(args.cropped, args.label_dir, args.label_out)


if __name__ == "__main__":
    main()
