"""YOLOv8n 단일 클래스(결함) 학습 + 평가.

COCO pretrained yolov8n을 증강 데이터셋으로 학습하고,
에포크별 validation 지표 / test 성능 / train-test 과적합 gap을 출력한다.

예)
    python train_eval.py --train <OR_TR_SR_경로> --valid <VALID_경로> --test <TEST_경로> --name case4_or_tr_sr
"""
import argparse
import os

import pandas as pd
import yaml
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True, help='학습 이미지 폴더 (YOLO 형식)')
    p.add_argument('--valid', required=True, help='검증 이미지 폴더')
    p.add_argument('--test', required=True, help='테스트 이미지 폴더')
    p.add_argument('--name', default='wire_defect_single_class', help='runs/detect/<name> 결과 폴더 이름')
    p.add_argument('--model', default='yolov8n.pt')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0', help="GPU 번호 또는 'cpu'")
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def f1(p, r):
    return 2 * p * r / (p + r + 1e-10) if (p > 0 or r > 0) else 0.0


def write_data_yaml(args, path='wire_data.yaml'):
    # 원본 데이터의 2개 클래스(끊어져서 생긴 결함 / 녹아서 생긴 결함)를 '결함' 1개 클래스로 통합
    data_config = {
        'train': args.train,
        'val': args.valid,
        'test': args.test,
        'names': {0: '결함'},
        'nc': 1,
    }
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True)
    return path


def print_epoch_metrics(csv_path):
    """results.csv에서 에포크별 validation 지표와 mAP50 최고 에포크 출력"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    p = df['metrics/precision(B)']
    r = df['metrics/recall(B)']
    table = pd.DataFrame({
        'epoch': df['epoch'].astype(int),
        'mAP50': df['metrics/mAP50(B)'],
        'mAP50-95': df['metrics/mAP50-95(B)'],
        'Precision': p,
        'Recall': r,
        'F1-score': [f1(pi, ri) for pi, ri in zip(p, r)],
    })

    print('=' * 100)
    print(f'에포크별 Validation 평가 지표 (총 {len(table)} epochs)')
    print('=' * 100)
    print(table.set_index('epoch').T.round(4).to_string())

    best = table.loc[table['mAP50'].idxmax()]
    print('\n' + '=' * 100)
    print(f"🏆 Validation mAP50 최고 에포크: {int(best['epoch'])}")
    print('=' * 100)
    for k in ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-score']:
        print(f'{k + ":":<11}{best[k]:.4f}')
    print('=' * 100 + '\n')


def print_metrics(title, res):
    """단일 클래스 평가 지표 출력"""
    mp, mr = res.box.mp, res.box.mr
    print(f"\n{'=' * 70}\n{title} Results (단일 클래스: 결함)\n{'=' * 70}")
    print(f'  mAP50-95:  {res.box.map:.4f}')
    print(f'  mAP50:     {res.box.map50:.4f}')
    print(f'  mAP75:     {res.box.map75:.4f}')
    print(f'  Precision: {mp:.4f}')
    print(f'  Recall:    {mr:.4f}')
    print(f'  F1-score:  {f1(mp, mr):.4f}')
    print('=' * 70 + '\n')


def main():
    args = parse_args()
    data_yaml = write_data_yaml(args)

    # ===== COCO pretrained 모델 로드 + 학습 =====
    model = YOLO(args.model)
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,  # 50 epoch 동안 개선 없으면 조기 종료
        save=True,
        project='runs/detect',
        name=args.name,
        exist_ok=False,  # 같은 이름이 있으면 name2, name3 ... 으로 저장
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=args.seed,
        device=args.device,
    )

    # ===== 실제 저장된 경로 기준으로 결과 읽기 =====
    save_dir = str(results.save_dir)
    print(f'\n📂 실제 저장 경로: {save_dir}\n')
    print_epoch_metrics(os.path.join(save_dir, 'results.csv'))

    # ===== best model 평가 =====
    best_model = YOLO(os.path.join(save_dir, 'weights', 'best.pt'))

    test_results = best_model.val(data=data_yaml, split='test')
    print_metrics('Test Dataset', test_results)

    # Train 데이터 평가 (과적합 확인용)
    train_results = best_model.val(data=data_yaml, split='train')
    print_metrics('Train Dataset', train_results)

    # ===== 과적합 분석 =====
    train_map50 = train_results.box.map50
    test_map50 = test_results.box.map50
    gap = train_map50 - test_map50
    print('=' * 70 + '\n과적합 분석 (Overfitting Analysis)\n' + '=' * 70)
    print(f'  Train mAP50: {train_map50:.4f}')
    print(f'  Test mAP50:  {test_map50:.4f}')
    print(f'  Gap:         {gap:.4f}')
    if gap < 0.05:
        print('  평가: ✅ 과적합 없음 (정상)')
    elif gap < 0.15:
        print('  평가: ⚠️ 경미한 과적합')
    else:
        print('  평가: 🔴 심한 과적합 - 정규화 필요')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
