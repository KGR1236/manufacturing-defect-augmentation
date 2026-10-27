# Detection — YOLOv8 결함 검출 학습/평가 및 시각화

전체 파이프라인(증강 → 학습/평가 → 시각화) 중 2·3단계. `data_augmentation`에서 만든
증강 데이터셋으로 YOLOv8n 결함 검출 모델을 학습하고, 증강 조합별 성능을 비교한다.

## 실험 설정

- **모델**: YOLOv8n (COCO pretrained `yolov8n.pt`)
- **클래스**: 원본의 2개 클래스(끊어져서 생긴 결함 / 녹아서 생긴 결함)를 `결함` 1개 클래스로 통합
- **하이퍼파라미터**: epochs 200, early stopping patience 50, imgsz 640, batch 16, optimizer auto, seed 42
- **검증/테스트 데이터**: 모든 Case 공통 (valid 100장, test 100장, 증강 미적용)

| Case | 학습 데이터 | 학습 이미지 수 |
|------|-------------|---------------:|
| 1 | OR (원본) | 300 |
| 2 | OR + TR (전통적 증강) | 1,200 |
| 3 | OR + SR (초해상도 증강) | 330 |
| 4 | OR + TR + SR | 1,320 |

## 사용법

```bash
pip install -r requirements.txt
```

1. **학습 + 평가** (Case마다 1회 실행)
   ```bash
   python train_eval.py --train <학습_경로> --valid <VALID_경로> --test <TEST_경로> --name case4_or_tr_sr
   ```
   - 에포크별 validation 지표(mAP50, mAP50-95, Precision, Recall, F1)와 mAP50 최고 에포크 출력
   - best 모델로 test / train 평가 후 train–test mAP50 gap(과적합) 출력
   - 결과는 `runs/detect/<name>/`에 저장 (`results.csv`, `weights/best.pt`)

2. **학습 곡선 비교 그래프**
   ```bash
   python plot_metrics.py \
     --run "OR=runs/detect/case1_or" \
     --run "OR+TR=runs/detect/case2_or_tr" \
     --run "OR+SR=runs/detect/case3_or_sr" \
     --run "OR+TR+SR=runs/detect/case4_or_tr_sr" \
     --out figures
   ```
   - 각 Case의 `results.csv`를 읽어 AP50 / Precision / Recall / F1-Score 그래프를 1장씩 저장
   - 초반 0 값은 결측 처리 후 보간하고, 이동평균(window 7)으로 추세를 표시

## 결과 (Test set)

| Case | 학습 데이터 | mAP50 | mAP50-95 | Precision | Recall | F1 |
|------|-------------|------:|---------:|----------:|-------:|---:|
| 1 | OR | 0.4322 | 0.1765 | 0.7352 | 0.4144 | 0.5301 |
| 2 | OR + TR | 0.5158 | **0.2136** | 0.6949 | 0.5225 | 0.5965 |
| 3 | OR + SR | 0.4180 | 0.1638 | 0.6530 | 0.4577 | 0.5382 |
| 4 | OR + TR + SR | **0.5372** | 0.2004 | 0.7265 | **0.5586** | **0.6316** |

학습 진행 (early stopping 기준):

| Case | 학습 에포크 | best 에포크 | Train mAP50 | Train–Test gap |
|------|-----------:|-----------:|------------:|---------------:|
| 1 | 173 | 123 | 0.9620 | 0.5298 |
| 2 | 122 | 72 | 0.9609 | 0.4451 |
| 3 | 117 | 67 | 0.9467 | 0.5286 |
| 4 | 143 | 93 | 0.9711 | 0.4339 |

- OR + TR + SR(Case 4)가 mAP50 · Recall · F1이 가장 높고, 원본 대비 mAP50 +0.105, F1 +0.102
- 전통적 증강(TR)이 포함된 Case 2, 4가 성능 향상 폭이 크며 train–test gap도 가장 작음
- SR만 추가한 Case 3(+30장)은 원본과 비슷한 수준으로, SR 단독 효과는 제한적
- 모든 Case에서 train mAP50이 0.95 안팎으로 test와 gap이 커서 과적합 경향이 있음
