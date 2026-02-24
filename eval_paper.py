"""
Paper evaluation script — evaluates trained SkateFormer on both
GTop test split and Omnilab real-world data.

Outputs: Accuracy, Precision, Recall, F1 for both test sets,
plus a ready-to-paste LaTeX table and per-class breakdown.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

ACTION_NAMES = [
    "brooming", "cleaningwindows", "downandgetup", "drinking",
    "fall-on-face", "inchairstandup", "walk-old-man", "push-object",
    "rugpull", "upbendfrmknees", "upfromground", "upbendfrmwaist", "walk",
]


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
                raise FileNotFoundError(f"No weights found in {work_dir}")

    print(f"  Loading weights: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    state = OrderedDict([[k.split("module.")[-1], v] for k, v in state.items()])
    model.load_state_dict(state)
    model.eval()
    model.cuda()
    return model, cfg


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


def evaluate_split(model, cfg, npz_path, x_key, y_key):
    """Run inference on a specific split from the NPZ and return metrics."""
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

    all_preds = []
    all_labels = []
    all_scores = []

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

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)

    acc = accuracy_score(labels, preds) * 100
    prec = precision_score(labels, preds, average="macro", zero_division=0) * 100
    rec = recall_score(labels, preds, average="macro", zero_division=0) * 100
    f1 = f1_score(labels, preds, average="macro", zero_division=0) * 100
    cm = confusion_matrix(labels, preds, labels=np.arange(len(ACTION_NAMES)))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "preds": preds,
        "labels": labels,
        "scores": scores,
        "report": classification_report(
            labels, preds, target_names=ACTION_NAMES, zero_division=0
        ),
    }


def print_results(name, res):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {res['accuracy']:.2f}%")
    print(f"  Precision: {res['precision']:.2f}%  (macro)")
    print(f"  Recall:    {res['recall']:.2f}%  (macro)")
    print(f"  F1-score:  {res['f1']:.2f}%  (macro)")
    print(f"\n  Classification Report:\n{res['report']}")


def print_latex(gtop_res, omni_res):
    print("\n" + "=" * 60)
    print("  LaTeX Table (copy-paste ready)")
    print("=" * 60)

    g = gtop_res
    o = omni_res

    print(r"""
\begin{table}[htbp]
  \centering
  \caption{Action recognition results of SkateFormer trained on GTop.
           The model is evaluated on both the GTop test split and the
           Omnilab real-world dataset.}
  \label{tab:action_all}
  \begin{tabular}{lcccc}
    \hline
    \textbf{Evaluation Set} & \textbf{Accuracy (\%%)} & \textbf{Precision (\%%)} & \textbf{Recall (\%%)} & \textbf{F1-score (\%%)} \\
    \hline""")
    print(f"    GTop test & {g['accuracy']:.2f} & {g['precision']:.2f} & {g['recall']:.2f} & {g['f1']:.2f} \\\\")
    print(f"    Omnilab   & {o['accuracy']:.2f} & {o['precision']:.2f} & {o['recall']:.2f} & {o['f1']:.2f} \\\\")
    print(r"""    \hline
  \end{tabular}
\end{table}""")


def save_results(result_dir, gtop_res, omni_res):
    os.makedirs(result_dir, exist_ok=True)

    summary = {}
    for name, res in [("gtop", gtop_res), ("omnilab", omni_res)]:
        summary[name] = {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "num_samples": len(res["labels"]),
        }

        csv_path = os.path.join(result_dir, f"predictions_{name}.csv")
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

    summary_path = os.path.join(result_dir, "paper_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {result_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Paper Evaluation")
    parser.add_argument("--config", type=str,
                        default="./config/test/synth_omni/SkateFormer_j_v2.yaml")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--result_dir", type=str,
                        default="/mnt/data_hdd/fzhi/eval/skate/result")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    print("=" * 60)
    print("  SkateFormer — Paper Evaluation")
    print("=" * 60)

    model, cfg = load_model(args.config, args.weights)
    npz_path = cfg["test_feeder_args"]["data_path"]

    # --- GTop test ---
    print("\n[1/2] Evaluating on GTop test split...")
    gtop_res = evaluate_split(model, cfg, npz_path,
                              "x_test_gtop", "y_test_gtop")
    if gtop_res:
        print_results("GTop Test (Synthetic held-out subjects)", gtop_res)
    else:
        print("  [WARN] x_test_gtop not found in NPZ, skipping")

    # --- Omnilab test ---
    print("\n[2/2] Evaluating on Omnilab real-world data...")
    omni_res = evaluate_split(model, cfg, npz_path,
                              "x_test_omni", "y_test_omni")
    if omni_res is None:
        omni_res = evaluate_split(model, cfg, npz_path, "x_test", "y_test")

    if omni_res:
        print_results("Omnilab (Real-world)", omni_res)

    # --- LaTeX & save ---
    if gtop_res and omni_res:
        print_latex(gtop_res, omni_res)
        save_results(args.result_dir, gtop_res, omni_res)

    print("\nDone!")


if __name__ == "__main__":
    main()
