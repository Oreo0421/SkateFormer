"""
从 main.py 保存的 each_class_acc.csv 中读取 confusion matrix 并绘图。

用法示例:
  # v5 CameraHMR (OmniLab test)
  python plot_confusion_matrix.py \
      --csv work_dir/synth_omni_v5_3d/SkateFormer_j/epoch1_test_each_class_acc.csv \
      --title "v5 3D (CameraHMR)" \
      --output /mnt/data_hdd/fzhi/eval/skateformer/img/cm_v5_camerahmr.png

  # v6 canonical (SAM-3D-Body)
  python plot_confusion_matrix.py \
      --csv work_dir/synth_omni_v6_canonical/SkateFormer_j/epoch1_test_each_class_acc.csv \
      --title "v6 Canonical (SAM-3D-Body)" \
      --output /mnt/data_hdd/fzhi/eval/skateformer/img/cm_v6_canonical.png
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


ACTION_NAMES = [
    "brooming", "cleaningwindows", "downandgetup", "drinking",
    "fall-on-face", "inchairstandup", "walk-old-man", "push-object",
    "rugpull", "upbendfrmknees", "upfromground", "upbendfrmwaist", "walk",
]

# 短标签（图内显示用）
ACTION_SHORT = [
    "broom", "cleanwin", "d&getup", "drink",
    "fall", "inchair", "oldwalk", "push",
    "rugpull", "upknees", "upground", "upwaist", "walk",
]


def load_cm_from_csv(csv_path):
    """读取 main.py 保存的 CSV: 第1行是 per-class acc，第2行起是混淆矩阵。"""
    with open(csv_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    # 第1行是 acc，跳过；第2行起是混淆矩阵
    cm_rows = []
    for line in lines[1:]:
        row = [int(x) for x in line.split(',')]
        cm_rows.append(row)

    cm = np.array(cm_rows, dtype=np.int64)
    return cm


def plot_confusion_matrix(cm, title, output_path, normalize=True, figsize=(11, 9)):
    num_classes = cm.shape[0]
    labels = ACTION_SHORT[:num_classes]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1  # 避免除零
        cm_plot = cm.astype(float) / row_sums
        fmt = '.2f'
        vmin, vmax = 0.0, 1.0
        cbar_label = 'Recall (row-normalized)'
    else:
        cm_plot = cm
        fmt = 'd'
        vmin, vmax = 0, cm.max()
        cbar_label = 'Count'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, interpolation='nearest', cmap='Blues',
                   vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=labels,
        yticklabels=labels,
        xlabel='Predicted label',
        ylabel='True label',
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    # 格子内写数字
    thresh = (vmin + vmax) / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_plot[i, j]
            text = f'{val:{fmt}}' if fmt != 'd' else f'{int(val)}'
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=7.5, color=color)

    # 标注整体 acc
    total = cm.sum()
    correct = np.diag(cm).sum()
    acc = correct / total * 100 if total > 0 else 0.0
    fig.text(0.5, 0.01, f'Overall Accuracy: {acc:.2f}%  ({correct}/{total})',
             ha='center', fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}  (acc={acc:.2f}%)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True,
                        help='Path to epoch*_test_each_class_acc.csv')
    parser.add_argument('--title', default='Confusion Matrix')
    parser.add_argument('--output', default='/mnt/data_hdd/fzhi/eval/skateformer/img/confusion_matrix.png')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Show raw counts instead of row-normalized values')
    args = parser.parse_args()

    cm = load_cm_from_csv(args.csv)
    print(f'Loaded confusion matrix: {cm.shape}')
    print(f'Total samples: {cm.sum()},  Correct: {np.diag(cm).sum()}')

    plot_confusion_matrix(
        cm,
        title=args.title,
        output_path=args.output,
        normalize=not args.no_normalize,
    )


if __name__ == '__main__':
    main()
