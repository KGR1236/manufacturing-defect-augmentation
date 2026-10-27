# Data Augmentation — GAN 기반 초해상도 결함 데이터 증강

전체 파이프라인(증강 → 학습/평가 → 시각화) 중 1단계. 원본(OR) 결함 이미지에
전통적 증강(회전/밝기/노이즈, TR)과 Real-ESRGAN 기반 초해상도 증강(SR)을
적용해 학습 데이터를 구성한다.

## 파이프라인

1. **전통적 증강 (OR → OR & TR)**
   ```bash
   python -m scripts.run_traditional_aug --source <OR_경로> --dest <OR_TR_경로>
   ```
2. **초해상도 증강 (OR & TR → SR)**
   ```bash
   python -m scripts.run_sr_conversion \
     --input <OR_TR_경로> --upscaled <임시_경로> --cropped <SR_경로> \
     --label-dir <OR_TR_경로> --label-out <SR_경로> \
     --model-path <RealESRGAN_x2plus.pth 경로> --crop-size 640
   ```
3. **QA 시각화** (선택)
   ```python
   from src.utils.visualize_labels import visualize_yolo_labels
   visualize_yolo_labels(image_dir, label_dir, num_samples=3)
   ```

> 모든 명령은 `data_augmentation/` 폴더에서 실행하세요.

## 설치

```bash
pip install -r requirements.txt
```

⚠️ torch는 CUDA 12.1 환경 기준입니다. 다른 CUDA 버전을 쓰는 경우
`requirements.txt`의 torch/torchvision 버전과 `--extra-index-url`을 맞춰 수정하세요.

## 출처

Real-ESRGAN 사전학습 모델(RealESRGAN_x2plus)은 별도로 다운로드해 `--model-path`로
지정해야 합니다. 결함 검출 데이터셋은 Roboflow Universe의 'FORMACION PROFESIONAL I'
와이어 로프 결함 검출 데이터셋을 기반으로 함.
