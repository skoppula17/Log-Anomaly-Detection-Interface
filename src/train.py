from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.hdfs import (
    load_hdfs_event_traces,
    load_hdfs_labels,
    build_vocab,
    encode_sequences,
    align_labels,
    make_next_event_windows,
    split_by_blockid,
)
from src.models.deeplog_lstm import DeepLogLSTM
from src.models.freq_baseline import build_freq_model
from src.eval import (
    evaluate_session_level,
    evaluate_baseline,
    sweep_topk_val,
    best_topk_by_f1,
    TOPK_SWEEP,
)


class NextEventDataset(Dataset):
    def __init__(self, windows: List[Tuple[List[int], int]]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        x, y = self.windows[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_traces", type=str, required=True)
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/cp1")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block_ids, seqs = load_hdfs_event_traces(args.event_traces)
    labels_map = load_hdfs_labels(args.labels)
    y_session = align_labels(block_ids, labels_map)

    vocab = build_vocab(seqs)
    enc = encode_sequences(seqs, vocab)

    train_idx, val_idx, test_idx = split_by_blockid(
        block_ids, y_session, train_ratio=0.7, val_ratio=0.15, seed=args.seed
    )
    splits = {
        "train": [block_ids[i] for i in train_idx],
        "val": [block_ids[i] for i in val_idx],
        "test": [block_ids[i] for i in test_idx],
    }
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))

    def gather(indices):
        return (
            [enc[i] for i in indices],
            y_session[indices],
            [block_ids[i] for i in indices],
        )

    train_seqs, y_train, train_bids = gather(train_idx)
    val_seqs, y_val, val_bids = gather(val_idx)
    test_seqs, y_test, test_bids = gather(test_idx)

    train_windows = []
    for s in train_seqs:
        train_windows.extend(make_next_event_windows(s, args.window))
    if len(train_windows) == 0:
        raise RuntimeError("No training windows produced. Try a smaller --window.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepLogLSTM(
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(
        NextEventDataset(train_windows),
        batch_size=256,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"epoch {ep}/{args.epochs}")
        running = 0.0
        n = 0
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
            pbar.set_postfix(loss=running / max(n, 1))

    meta = {
        "window": args.window,
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "vocab": vocab,
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")

    val_sweep = sweep_topk_val(model, device, val_seqs, y_val, args.window)
    with open(out_dir / "val_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topk", "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(val_sweep)

    best_topk = best_topk_by_f1(val_sweep)
    print(f"Best topk on VAL (by F1): {best_topk}")

    lstm_metrics, cm_path = evaluate_session_level(
        model=model,
        device=device,
        test_block_ids=test_bids,
        test_sequences=test_seqs,
        y_true_session=y_test,
        window=args.window,
        topk=best_topk,
        out_dir=out_dir,
    )

    top_by_last, global_top = build_freq_model(train_seqs, args.window)
    baseline_metrics = evaluate_baseline(
        top_by_last,
        global_top,
        test_seqs,
        y_test,
        args.window,
        best_topk,
        out_dir=out_dir,
    )

    print("\nLSTM (best topk=%d) on TEST:" % best_topk)
    for k, v in lstm_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Baseline on TEST:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
