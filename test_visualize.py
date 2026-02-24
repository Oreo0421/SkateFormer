"""
SkateFormer Test with Visualization
- Evaluates on GTop test + Omnilab test
- Skeleton images  -> /mnt/data_hdd/fzhi/eval/skate/image/
- Result data      -> /mnt/data_hdd/fzhi/eval/skate/result/
- Summary printed to terminal
"""

import os
import sys
import csv
import json
import argparse
from collections import OrderedDict

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
)

# ─────────────── Config ───────────────

IMAGE_DIR = "/mnt/data_hdd/fzhi/eval/skate/image"
RESULT_DIR = "/mnt/data_hdd/fzhi/eval/skate/result"

ACTION_NAMES = [
    "brooming", "cleaningwindows", "downandgetup", "drinking",
    "fall-on-face", "inchairstandup", "walk-old-man", "push-object",
    "rugpull", "upbendfrmknees", "upfromground", "upbendfrmwaist", "walk",
]

BONES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
    (0, 6), (1, 7), (6, 7), (6, 8), (8, 10), (7, 9), (9, 11),
]

LEFT_JOINTS = {0, 2, 4, 6, 8, 10}


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition(".")
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def load_model(config_path, weights_override=None):
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    Model = import_class(cfg["model"])
    model = Model(**cfg["model_args"])

    work_dir = cfg.get("work_dir", ".")
    weights_path = weights_override or cfg.get("weights")

    if not weights_path or not os.path.exists(weights_path):
        if weights_path and not os.path.exists(weights_path):
            print(f"  [WARN] Configured weights not found: {weights_path}")
        best_pt = os.path.join(work_dir, "best_model.pt")
        if os.path.exists(best_pt):
            weights_path = best_pt
        else:
            pts = sorted(
                [f for f in os.listdir(work_dir)
                 if f.startswith("runs-") and f.endswith(".pt")],
                key=lambda x: int(x.split("-")[1]),
            )
            if pts:
                weights_path = os.path.join(work_dir, pts[-1])
            else:
                raise FileNotFoundError(f"No weights in {work_dir}")

    print(f"  Loading weights: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    state = OrderedDict([[k.split("module.")[-1], v] for k, v in state.items()])
    model.load_state_dict(state)
    model.eval()
    model.cuda()
    return model, cfg


def get_raw_skeleton(npz_path, x_key="x_test", y_key="y_test"):
    """Load raw skeleton data for visualization."""
    npz = np.load(npz_path, allow_pickle=True)
    if x_key not in npz:
        return None, None
    x = npz[x_key]
    y = np.where(npz[y_key] > 0)[1]
    N, T, feat = x.shape
    V = 12
    D = feat // V
    x = x.reshape(N, T, V, D)
    if D == 2:
        pad = np.zeros((N, T, V, 1), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=-1)
    return x, y


def _normalize_data_for_feeder(x_data, num_point=12, num_dim=3):
    """Reshape raw NPZ (N,T,V*dim) to feeder format (N,C,T,V,M)."""
    x_data = np.array(x_data)
    if x_data.ndim == 3:
        N, T, D = x_data.shape
        if D == num_point * 3:
            x_data = x_data.reshape(N, T, num_point, 3)
        elif D == num_point * 2:
            x_data = x_data.reshape(N, T, num_point, 2)
            pad = np.zeros((N, T, num_point, 1), dtype=x_data.dtype)
            x_data = np.concatenate([x_data, pad], axis=-1)
        else:
            raise ValueError(f"Unsupported x shape {x_data.shape}, D={D}")
        x_data = x_data.transpose(0, 3, 1, 2)[:, :, :, :, None]
    return x_data


def run_inference(model, cfg, npz_path, x_key, y_key):
    """Run inference on a specific split."""
    npz = np.load(npz_path, allow_pickle=True)
    if x_key not in npz:
        return None

    x_data = npz[x_key]
    y_data = npz[y_key]
    true_labels = np.where(y_data > 0)[1]

    num_point = cfg.get("model_args", {}).get("num_points", 12)
    num_dim = cfg.get("test_feeder_args", {}).get("num_dim", 3)
    x_data = _normalize_data_for_feeder(x_data, num_point=num_point, num_dim=num_dim)

    Feeder = import_class(cfg["feeder"])
    test_args = dict(cfg["test_feeder_args"])
    test_args["data_path"] = npz_path
    dataset = Feeder(**test_args)
    dataset.data = x_data
    dataset.label = true_labels

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.get("test_batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("num_worker", 2),
        drop_last=False,
    )

    all_preds, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for data, index_t, label, index in loader:
            data = data.float().cuda()
            index_t = index_t.float().cuda()
            output = model(data, index_t)
            probs = torch.softmax(output, dim=1)
            _, pred = torch.max(output, 1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.numpy())
            all_scores.append(probs.cpu().numpy())

    return {
        "preds": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
        "scores": np.concatenate(all_scores),
    }


def compute_metrics(res):
    """Compute all metrics from inference results."""
    preds, labels = res["preds"], res["labels"]
    cm = confusion_matrix(labels, preds, labels=np.arange(len(ACTION_NAMES)))
    diag = np.diag(cm)
    row_sum = np.sum(cm, axis=1)
    class_accs = np.where(row_sum > 0, diag / row_sum, 0.0)
    overall_acc = np.mean(preds == labels)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "overall_acc": overall_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "class_accs": class_accs,
        "cm": cm,
        "diag": diag,
        "row_sum": row_sum,
    }


def count_valid_frames(seq):
    return int(np.sum(np.any(seq != 0, axis=(1, 2))))


def draw_skeleton_frame(ax, joints_2d, title=None):
    valid = joints_2d[np.any(joints_2d != 0, axis=1)]
    if len(valid) > 0:
        margin = 0.3
        xmin, xmax = valid[:, 0].min() - margin, valid[:, 0].max() + margin
        ymin, ymax = valid[:, 1].min() - margin, valid[:, 1].max() + margin
    else:
        xmin, xmax, ymin, ymax = -2, 2, -2, 2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, pad=2)

    for j1, j2 in BONES:
        x_vals = [joints_2d[j1, 0], joints_2d[j2, 0]]
        y_vals = [joints_2d[j1, 1], joints_2d[j2, 1]]
        if np.all(np.array(x_vals) == 0) and np.all(np.array(y_vals) == 0):
            continue
        ax.plot(x_vals, y_vals, "-", color="#555555", linewidth=1.5, zorder=1)

    for j_idx in range(joints_2d.shape[0]):
        x, y = joints_2d[j_idx]
        if x == 0 and y == 0:
            continue
        c = "#2196F3" if j_idx in LEFT_JOINTS else "#F44336"
        ax.plot(x, y, "o", color=c, markersize=4, zorder=2)


def draw_sequence(raw_seq, true_label, pred_label, pred_conf, sample_idx, save_path):
    valid_t = count_valid_frames(raw_seq)
    if valid_t == 0:
        valid_t = 1
    n_frames = min(8, valid_t)
    frame_indices = np.linspace(0, valid_t - 1, n_frames, dtype=int)

    is_correct = true_label == pred_label
    border_color = "#4CAF50" if is_correct else "#F44336"
    status = "CORRECT" if is_correct else "WRONG"

    fig = plt.figure(figsize=(n_frames * 1.8, 3.2))
    fig.suptitle(
        f"#{sample_idx}  True: {ACTION_NAMES[true_label]}  |  "
        f"Pred: {ACTION_NAMES[pred_label]} ({pred_conf:.1%})  |  {status}",
        fontsize=10, fontweight="bold", color=border_color, y=0.98,
    )

    gs = gridspec.GridSpec(1, n_frames, wspace=0.05)
    for i, fi in enumerate(frame_indices):
        ax = fig.add_subplot(gs[0, i])
        draw_skeleton_frame(ax, raw_seq[fi, :, :2], title=f"t={fi}")

    fig.patch.set_linewidth(3)
    fig.patch.set_edgecolor(border_color)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", pad_inches=0.15,
                facecolor="white", edgecolor=border_color)
    plt.close(fig)


def draw_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    tick_marks = np.arange(len(ACTION_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(ACTION_NAMES, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_per_class_accuracy(class_accs, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ACTION_NAMES))
    colors = ["#4CAF50" if a >= 0.5 else "#FF9800" if a >= 0.2 else "#F44336"
              for a in class_accs]
    bars = ax.bar(x, class_accs * 100, color=colors, edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars, class_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=np.mean(class_accs) * 100, color="#666", linestyle="--",
               linewidth=1, label=f"Mean: {np.mean(class_accs):.1%}")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_split_results(split_name, res, metrics):
    preds, labels, scores = res["preds"], res["labels"], res["scores"]
    m = metrics

    print(f"\n{'=' * 70}")
    print(f"  {split_name}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {m['overall_acc']:.2%}  ({int(np.sum(m['diag']))}/{len(labels)})")
    print(f"  Precision: {m['precision']:.2%}  (macro)")
    print(f"  Recall:    {m['recall']:.2%}  (macro)")
    print(f"  F1-score:  {m['f1']:.2%}  (macro)\n")

    print(f"  {'Class':<20s} {'Acc':>8s}  {'Correct':>8s}  {'Total':>8s}")
    print("  " + "-" * 50)
    for i, name in enumerate(ACTION_NAMES):
        total = m["row_sum"][i]
        correct = m["diag"][i]
        acc = m["class_accs"][i]
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {name:<20s} {acc:>7.1%}  {correct:>8d}  {total:>8d}  {bar}")
    print("  " + "-" * 50)
    print(f"  {'MEAN':<20s} {np.mean(m['class_accs']):>7.1%}\n")

    print("  Per-Sample Predictions:")
    print(f"  {'#':>4s}  {'True Label':<20s}  {'Predicted':<20s}  {'Conf':>6s}  {'':>7s}")
    print("  " + "-" * 65)
    for i in range(len(labels)):
        tl, pl = labels[i], preds[i]
        conf = scores[i, pl]
        status = "  OK" if tl == pl else "  MISS"
        marker = "" if tl == pl else " <<"
        print(f"  {i:>4d}  {ACTION_NAMES[tl]:<20s}  {ACTION_NAMES[pl]:<20s}  "
              f"{conf:>5.1%} {status}{marker}")
    print()


def save_split_results(result_dir, split_tag, res, metrics):
    csv_path = os.path.join(result_dir, f"predictions_{split_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "true_label", "true_name", "pred_label",
                         "pred_name", "confidence", "correct"])
        for i in range(len(res["labels"])):
            tl, pl = res["labels"][i], res["preds"][i]
            writer.writerow([
                i, tl, ACTION_NAMES[tl], pl, ACTION_NAMES[pl],
                f"{res['scores'][i, pl]:.4f}", int(tl == pl),
            ])
    print(f"  -> {csv_path}")

    cls_path = os.path.join(result_dir, f"per_class_accuracy_{split_tag}.csv")
    with open(cls_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "accuracy", "correct", "total"])
        for i, name in enumerate(ACTION_NAMES):
            writer.writerow([i, name, f"{metrics['class_accs'][i]:.4f}",
                             int(metrics["diag"][i]), int(metrics["row_sum"][i])])
    print(f"  -> {cls_path}")


def process_split(split_name, split_tag, model, cfg, npz_path,
                  x_key, y_key, image_dir, result_dir):
    """Full evaluation + visualization for one test split."""
    print(f"\n{'─' * 70}")
    print(f"  Processing: {split_name}")
    print(f"{'─' * 70}")

    res = run_inference(model, cfg, npz_path, x_key, y_key)
    if res is None:
        print(f"  [SKIP] {x_key} not found in NPZ")
        return None, None

    metrics = compute_metrics(res)
    print_split_results(split_name, res, metrics)

    # Skeleton images
    raw_x, raw_y = get_raw_skeleton(npz_path, x_key, y_key)
    if raw_x is not None:
        skel_dir = os.path.join(image_dir, f"skeletons_{split_tag}")
        os.makedirs(skel_dir, exist_ok=True)
        for i in range(len(res["labels"])):
            tl, pl = res["labels"][i], res["preds"][i]
            conf = res["scores"][i, pl]
            status = "correct" if tl == pl else "wrong"
            fname = f"{i:03d}_{ACTION_NAMES[tl]}_{status}.png"
            draw_sequence(raw_x[i], tl, pl, conf, i, os.path.join(skel_dir, fname))
        print(f"  -> {len(res['labels'])} skeleton images saved to {skel_dir}/")

    # Confusion matrix
    cm_path = os.path.join(image_dir, f"confusion_matrix_{split_tag}.png")
    draw_confusion_matrix(metrics["cm"], f"Confusion Matrix ({split_name})", cm_path)
    print(f"  -> {cm_path}")

    # Per-class accuracy chart
    bar_path = os.path.join(image_dir, f"per_class_accuracy_{split_tag}.png")
    draw_per_class_accuracy(metrics["class_accs"],
                            f"Per-Class Accuracy ({split_name})", bar_path)
    print(f"  -> {bar_path}")

    # CSV results
    save_split_results(result_dir, split_tag, res, metrics)

    return res, metrics


def main():
    parser = argparse.ArgumentParser(description="SkateFormer Test + Visualization")
    parser.add_argument("--config", type=str,
                        default="./config/test/synth_omni/SkateFormer_j_v2.yaml")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    torch.cuda.set_device(args.device)

    print("=" * 70)
    print("  SkateFormer Test with Visualization")
    print("=" * 70)

    model, cfg = load_model(args.config, args.weights)
    npz_path = cfg["test_feeder_args"]["data_path"]

    splits = [
        ("GTop Test (Synthetic held-out)", "gtop", "x_test_gtop", "y_test_gtop"),
        ("Omnilab (Real-world)", "omnilab", "x_test_omni", "y_test_omni"),
    ]

    all_results = {}
    for split_name, split_tag, x_key, y_key in splits:
        res, metrics = process_split(
            split_name, split_tag, model, cfg, npz_path,
            x_key, y_key, args.image_dir, args.result_dir,
        )
        if res is not None:
            all_results[split_tag] = {"res": res, "metrics": metrics}

    # Fallback: if no gtop/omni keys, try x_test/y_test
    if not all_results:
        res, metrics = process_split(
            "Test Set", "test", model, cfg, npz_path,
            "x_test", "y_test", args.image_dir, args.result_dir,
        )
        if res is not None:
            all_results["test"] = {"res": res, "metrics": metrics}

    # Save summary
    summary = {}
    for tag, data in all_results.items():
        m = data["metrics"]
        summary[tag] = {
            "accuracy": float(m["overall_acc"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "mean_class_accuracy": float(np.mean(m["class_accs"])),
            "num_samples": int(len(data["res"]["labels"])),
        }

    summary_path = os.path.join(args.result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  -> {summary_path}")

    # Final summary
    print("\n" + "=" * 70)
    for tag, s in summary.items():
        print(f"  {tag:>10s}:  Acc={s['accuracy']:.2%}  "
              f"Prec={s['precision']:.2%}  Rec={s['recall']:.2%}  "
              f"F1={s['f1']:.2%}  ({s['num_samples']} samples)")
    print(f"\n  Images:  {args.image_dir}")
    print(f"  Results: {args.result_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
