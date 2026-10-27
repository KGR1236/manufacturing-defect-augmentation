"""Real-ESRGAN x2plus 기반 초해상도(SR) 변환 (포스터 [그림 2])"""
import os
from pathlib import Path

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def build_upsampler(model_path: str, tile: int = 0) -> RealESRGANer:
    """RealESRGAN_x2plus 모델 로드"""
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RealESRGANer(
        scale=2,
        model_path=model_path,
        model=model,
        tile=tile,  # 메모리 부족 시 400 정도로 조정
        tile_pad=10,
        pre_pad=0,
        half=(device == "cuda"),
        device=device,
    )


def upscale_images(input_dir: str, output_dir: str, model_path: str, suffix: str = "_s") -> None:
    """input_dir의 jpg 이미지를 x2 업스케일하여 output_dir에 저장. 파일명에 suffix를 붙여 SR 이미지임을 표시."""
    os.makedirs(output_dir, exist_ok=True)
    upsampler = build_upsampler(model_path)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg"))]
    print(f"총 {len(image_files)}개의 이미지를 처리합니다. 디바이스: {upsampler.device}")

    for idx, img_name in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, img_name)
        stem = Path(img_name).stem
        ext = Path(img_name).suffix
        output_path = os.path.join(output_dir, f"{stem}{suffix}{ext}")

        try:
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[{idx}/{len(image_files)}] 오류: {img_name} 읽기 실패")
                continue

            output, _ = upsampler.enhance(img, outscale=2)
            cv2.imwrite(output_path, output)
            print(
                f"[{idx}/{len(image_files)}] 완료: {img_name} "
                f"({img.shape[1]}x{img.shape[0]} → {output.shape[1]}x{output.shape[0]})"
            )
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] 오류: {img_name} - {e}")

    print("\n모든 이미지 처리 완료!")
