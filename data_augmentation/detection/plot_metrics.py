"""Case별 학습 곡선(AP50 / Precision / Recall / F1-Score) 비교 그래프.

각 Case 학습 결과 폴더의 results.csv(에포크별 validation 지표)를 읽어
지표마다 그래프 1장씩 저장한다.

예)
    python plot_metrics.py \
      --run "OR=runs/detect/case1_or" \
      --run "OR+TR=runs/detect/case2_or_tr" \
      --run "OR+SR=runs/detect/case3_or_sr" \
      --run "OR+TR+SR=runs/detect/case4_or_tr_sr" \
      --out figures
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
METRICS = ['AP50', 'Precision', 'Recall', 'F1-Score']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True,
                   help='"라벨=결과폴더" 형식, 여러 번 지정 (결과폴더 안의 results.csv를 읽음)')
    p.add_argument('--out', default='figures', help='그래프 저장 폴더')
    p.add_argument('--smooth', type=int, default=7, help='이동평균 window')
    p.add_argument('--xmax', type=int, default=200)
    p.add_argument('--ylim', type=float, nargs=2, default=[0.5, 1.0])
    return p.parse_args()


def load_run(run_dir):
    df = pd.read_csv(os.path.join(run_dir, 'results.csv'))
    df.columns = df.columns.str.strip()
    p = df['metrics/precision(B)']
    r = df['metrics/recall(B)']
    return pd.DataFrame({
        'epoch': df['epoch'].astype(int),
        'AP50': df['metrics/mAP50(B)'],
        'Precision': p,
        'Recall': r,
        'F1-Score': 2 * p * r / (p + r + 1e-10),
    })


def preprocess_for_plot(df, metric, smooth_window=7):
    s = pd.to_numeric(df[metric], errors='coerce')
    # 0 값(초반 에포크의 계산 실패/미측정)은 스파이크가 되므로 결측 처리 후 보간
    s = s.mask(s <= 0, np.nan)
    s = s.interpolate(limit_direction='both')
    # 이동 평균으로 추세 강조
    return s.rolling(window=smooth_window, min_periods=1).mean()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    runs = {}
    for item in args.run:
        label, run_dir = item.split('=', 1)
        runs[label] = load_run(run_dir)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for color, (label, df) in zip(COLORS, runs.items()):
            smooth = preprocess_for_plot(df, metric, args.smooth)
            ax.plot(df['epoch'], smooth, color=color, linewidth=2.5, label=label, alpha=0.95)
            # 마지막 epoch 값 표시
            ax.scatter(df['epoch'].iloc[-1], smooth.iloc[-1], color=color, s=25, alpha=0.9)

        ax.set_title(metric, fontsize=20, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=20, fontweight='bold')
        ax.set_ylabel(metric, fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc='lower right', fontsize=20)
        ax.set_xlim(0, args.xmax)
        ax.set_ylim(*args.ylim)
        ax.tick_params(axis='both', which='major', labelsize=15)

        plt.tight_layout()
        path = os.path.join(args.out, f'training_metric_{metric}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'✅ {path}')


if __name__ == '__main__':
    main()
