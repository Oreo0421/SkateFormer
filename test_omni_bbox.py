"""
Test OmniLab: bbox裁剪 + 人工标注keypoints → SkateFormer v4

将人工标注的keypoints平移到裁剪图坐标系 (x-bbox_x, y-bbox_y)，
然后skeleton-centric归一化，用v4 ckpt测试。
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
from glob import glob
from collections import OrderedDict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

COCO_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NUM_JOINTS = 12
T_MAX = 300
IDX_L_SHOULDER, IDX_R_SHOULDER = 0, 1
IDX_L_HIP, IDX_R_HIP = 6, 7

ACTION_MAP_OMNI = {
    "brooming": 0, "cleaningwindows": 1, "downandgetup": 2,
    "drinking": 3, "fall-on-face": 4, "inchairstandup": 5,
    "walk-old-man": 6, "push-object": 7, "rugpull": 8,
    "upbendfrmknees": 9, "upfromground": 10, "upbendfrmwaist": 11, "walk": 12,
}

ACTION_NAMES = [
    "brooming", "cleaningwindows", "downandgetup", "drinking",
    "fall-on-face", "inchairstandup", "walk-old-man", "push-object",
    "rugpull", "upbendfrmknees", "upfromground", "upbendfrmwaist", "walk",
]

NUM_CLASSES = 13
POSSIBLE_PERSONS = ["aileen", "jingrui", "lars", "roman", "zexin"]


def normalize_skeleton(joints, ndim=2):
    coords = joints[:, :, :ndim].copy()
    hip_center = (coords[:, IDX_L_HIP:IDX_L_HIP+1, :] +
                  coords[:, IDX_R_HIP:IDX_R_HIP+1, :]) / 2.0
    shoulder_center = (coords[:, IDX_L_SHOULDER:IDX_L_SHOULDER+1, :] +
                       coords[:, IDX_R_SHOULDER:IDX_R_SHOULDER+1, :]) / 2.0
    torso_vec = shoulder_center - hip_center
    torso_len = np.linalg.norm(torso_vec, axis=-1)
    valid_mask = torso_len.flatten() > 1e-6
    scale = float(np.median(torso_len.flatten()[valid_mask])) if valid_mask.any() else 1.0
    scale = max(scale, 1e-6)
    centered = coords - hip_center
    normalized = centered / scale
    result = joints.copy()
    result[:, :, :ndim] = normalized
    return result


def process_omnilab_bbox_crop(ann_root, out_dim=3):
    """人工标注keypoints + bbox裁剪（坐标平移到裁剪图空间）"""
    all_sequences = []
    all_labels = []
    skipped = 0

    json_files = sorted(glob(os.path.join(ann_root, "*.json")))

    for jf in json_files:
        basename = os.path.splitext(os.path.basename(jf))[0]
        omni_action = None
        for p in POSSIBLE_PERSONS:
            idx = basename.find(f"_{p}_")
            if idx != -1:
                omni_action = basename[:idx]
                break
        if omni_action is None or omni_action not in ACTION_MAP_OMNI:
            skipped += 1
            continue

        label = ACTION_MAP_OMNI[omni_action]
        with open(jf, "r") as f:
            coco = json.load(f)

        ann_by_frame = {}
        for ann in coco["annotations"]:
            ann_by_frame[ann["image_id"]] = ann

        num_frames = len(coco["images"])
        if num_frames == 0:
            skipped += 1
            continue

        frame_ids = sorted(ann_by_frame.keys())
        joints_seq = np.zeros((num_frames, NUM_JOINTS, out_dim), dtype=np.float32)

        for fi, frame_id in enumerate(frame_ids):
            ann = ann_by_frame[frame_id]
            kps = np.array(ann["keypoints"]).reshape(17, 3)
            bbox = ann.get("bbox", [0, 0, 1, 1])
            bx, by = float(bbox[0]), float(bbox[1])

            for j_idx, coco_idx in enumerate(COCO_INDICES):
                x, y, vis = kps[coco_idx]
                if vis > 0:
                    # 平移到裁剪图坐标系
                    joints_seq[fi, j_idx, 0] = x - bx
                    joints_seq[fi, j_idx, 1] = y - by

        normed = normalize_skeleton(joints_seq, ndim=2)
        all_sequences.append(normed)
        all_labels.append(label)

    if skipped:
        print(f"  [INFO] Skipped {skipped} non-matching annotations")
    return all_sequences, all_labels


def pad_sequences(sequences, t_max, dim):
    N = len(sequences)
    padded = np.zeros((N, t_max, NUM_JOINTS * dim), dtype=np.float32)
    for i, seq in enumerate(sequences):
        T = min(seq.shape[0], t_max)
        flat = seq[:T].reshape(T, -1)
        padded[i, :T] = flat
    return padded


def make_onehot(labels, num_classes):
    N = len(labels)
    onehot = np.zeros((N, num_classes), dtype=np.float32)
    for i, l in enumerate(labels):
        onehot[i, l] = 1.0
    return onehot


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
        weights_path = os.path.join(work_dir, "best_model.pt")
    print(f"  Loading weights: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    state = OrderedDict([[k.split("module.")[-1], v] for k, v in state.items()])
    model.load_state_dict(state)
    model.eval()
    model.cuda()
    return model, cfg


def normalize_data_for_feeder(x_data, num_point=12):
    x_data = np.array(x_data)
    if x_data.ndim == 3:
        N, T, D = x_data.shape
        if D == num_point * 3:
            x_data = x_data.reshape(N, T, num_point, 3)
        elif D == num_point * 2:
            x_data = x_data.reshape(N, T, num_point, 2)
            pad = np.zeros((N, T, num_point, 1), dtype=x_data.dtype)
            x_data = np.concatenate([x_data, pad], axis=-1)
        x_data = x_data.transpose(0, 3, 1, 2)[:, :, :, :, None]
    return x_data


def evaluate_data(model, cfg, x_data, y_onehot, split_name="test"):
    true_labels = np.where(y_onehot > 0)[1]
    num_point = cfg.get("model_args", {}).get("num_points", 12)
    x_data = normalize_data_for_feeder(x_data, num_point=num_point)

    Feeder = import_class(cfg["feeder"])
    test_args = dict(cfg["test_feeder_args"])
    dataset = Feeder(**test_args)
    dataset.data = x_data
    dataset.label = true_labels
    dataset.sample_name = [f"{split_name}_{i}" for i in range(len(true_labels))]

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.get("test_batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("num_worker", 2),
        drop_last=False,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, index_t, label, index in loader:
            data = data.float().cuda()
            index_t = index_t.float().cuda()
            output = model(data, index_t)
            _, pred = torch.max(output, 1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = accuracy_score(labels, preds) * 100
    prec = precision_score(labels, preds, average="macro", zero_division=0) * 100
    rec = recall_score(labels, preds, average="macro", zero_division=0) * 100
    f1 = f1_score(labels, preds, average="macro", zero_division=0) * 100

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "num_samples": len(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/test/synth_omni/SkateFormer_j_v4.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--ann_root",
                        default="/mnt/dst_datasets/own_omni_dataset/action16_2022/annotations_final")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    print("=" * 70)
    print("  OmniLab BBox裁剪 + 人工标注 keypoints → SkateFormer v4")
    print("=" * 70)

    model, cfg = load_model(args.config, args.weights)
    npz_path = cfg["test_feeder_args"]["data_path"]

    # GTop (unchanged)
    print("\n[1/2] GTop test (from NPZ)...")
    npz = np.load(npz_path, allow_pickle=True)
    gtop_res = evaluate_data(model, cfg, npz["x_test_gtop"], npz["y_test_gtop"], "gtop")

    # OmniLab bbox crop + manual keypoints
    print("\n[2/2] OmniLab — bbox裁剪 + 人工标注...")
    omni_seqs, omni_labels = process_omnilab_bbox_crop(args.ann_root, out_dim=3)
    print(f"  -> {len(omni_seqs)} sequences")
    x_omni = pad_sequences(omni_seqs, T_MAX, dim=3)
    y_omni = make_onehot(omni_labels, NUM_CLASSES)
    omni_res = evaluate_data(model, cfg, x_omni, y_omni, "omni_bbox")

    # Results
    print(f"\n{'=' * 70}")
    print(f"  gtop:     Acc={gtop_res['accuracy']:.2f}%  Prec={gtop_res['precision']:.2f}%  "
          f"Rec={gtop_res['recall']:.2f}%  F1={gtop_res['f1']:.2f}%  ({gtop_res['num_samples']} samples)")
    print(f"  omnilab:  Acc={omni_res['accuracy']:.2f}%  Prec={omni_res['precision']:.2f}%  "
          f"Rec={omni_res['recall']:.2f}%  F1={omni_res['f1']:.2f}%  ({omni_res['num_samples']} samples)")
    print(f"  (bbox裁剪 + 人工标注keypoints)")
    print("=" * 70)


if __name__ == "__main__":
    main()
