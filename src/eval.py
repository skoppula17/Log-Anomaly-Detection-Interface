from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

from src.models.deeplog_lstm import DeepLogLSTM
from src.models.freq_baseline import build_freq_model, session_is_anomalous_freq
from src.data.hdfs import load_hdfs_event_traces, load_hdfs_labels, encode_sequences, align_labels, make_next_event_windows

TOPK_SWEEP = [1, 3, 5, 7, 9]


def _predict_next_topk(model: torch.nn.Module, device: str, window_tokens: List[int], topk: int) -> List[int]:
    x = torch.tensor(window_tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        tk = torch.topk(logits, k=topk, dim=-1).indices.squeeze(0).tolist()
    return tk


def session_is_anomalous(model: torch.nn.Module, device: str, seq: List[int], window: int, topk: int) -> int:
    windows = make_next_event_windows(seq, window)
    if not windows:
        return 0
    for xw, y_true in windows:
        preds = _predict_next_topk(model, device, xw, topk=topk)
        if y_true not in preds:
            return 1
    return 0


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def evaluate_session_level(
    model: torch.nn.Module,
    device: str,
    test_block_ids: List[str],
    test_sequences: List[List[int]],
    y_true_session: np.ndarray,
    window: int,
    topk: int,
    out_dir: Optional[Path] = None,
) -> Tuple[Dict[str, float], Optional[str]]:
    model.eval()
    y_pred = np.zeros(len(test_sequences), dtype=np.int64)
    for i, seq in enumerate(test_sequences):
        y_pred[i] = session_is_anomalous(model, device, seq, window=window, topk=topk)
    metrics = _compute_metrics(y_true_session, y_pred)
    cm_path = None
    if out_dir:
        cm = confusion_matrix(y_true_session, y_pred)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cm)
        ax.set_title("Confusion Matrix (session-level)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (r, c), val in np.ndenumerate(cm):
            ax.text(c, r, str(val), ha="center", va="center")
        cm_path = str(out_dir / "confusion_matrix.png")
        fig.tight_layout()
        fig.savefig(cm_path, dpi=160)
        plt.close(fig)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics, cm_path


def evaluate_baseline(
    top_by_last: Dict,
    global_top: List[Tuple[int, int]],
    test_sequences: List[List[int]],
    y_true_session: np.ndarray,
    window: int,
    topk: int,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    y_pred = np.zeros(len(test_sequences), dtype=np.int64)
    for i, seq in enumerate(test_sequences):
        y_pred[i] = session_is_anomalous_freq(
            top_by_last, global_top, seq, window=window, topk=topk
        )
    metrics = _compute_metrics(y_true_session, y_pred)
    if out_dir:
        (out_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def sweep_topk_val(
    model: torch.nn.Module,
    device: str,
    val_sequences: List[List[int]],
    y_val: np.ndarray,
    window: int,
) -> List[Dict[str, float]]:
    rows = []
    for topk in TOPK_SWEEP:
        y_pred = np.zeros(len(val_sequences), dtype=np.int64)
        for i, seq in enumerate(val_sequences):
            y_pred[i] = session_is_anomalous(model, device, seq, window=window, topk=topk)
        m = _compute_metrics(y_val, y_pred)
        rows.append({"topk": topk, **m})
    return rows


def best_topk_by_f1(rows: List[Dict[str, float]]) -> int:
    return max(rows, key=lambda r: r["f1"])["topk"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--meta", type=str, required=True)
    ap.add_argument("--event_traces", type=str, required=True)
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--topk", type=int, default=9)
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    vocab = meta["vocab"]
    window = int(meta["window"])
    emb_dim = int(meta["emb_dim"])
    hidden_dim = int(meta["hidden_dim"])

    block_ids, seqs = load_hdfs_event_traces(args.event_traces)
    labels_map = load_hdfs_labels(args.labels)
    y_session = align_labels(block_ids, labels_map)

    enc = encode_sequences(seqs, vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepLogLSTM(vocab_size=len(vocab), emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    out_dir = Path(args.ckpt).parent
    metrics, cm_path = evaluate_session_level(
        model=model,
        device=device,
        test_block_ids=block_ids,
        test_sequences=enc,
        y_true_session=y_session,
        window=window,
        topk=args.topk,
        out_dir=out_dir,
    )

    print("Session-level metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Saved confusion matrix plot to: {cm_path}")

if __name__ == "__main__":
    main()
