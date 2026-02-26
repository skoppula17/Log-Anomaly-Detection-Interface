from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.hdfs import (
    load_hdfs_event_traces,
    load_hdfs_labels,
    build_vocab,
    encode_sequences,
    align_labels,
    make_next_event_windows,
)
from src.models.deeplog_lstm import DeepLogLSTM
from src.eval import evaluate_session_level

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
    ap.add_argument("--topk", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block_ids, seqs = load_hdfs_event_traces(args.event_traces)
    labels_map = load_hdfs_labels(args.labels)
    y_session = align_labels(block_ids, labels_map)

    vocab = build_vocab(seqs)
    enc = encode_sequences(seqs, vocab)

    # Train/test split at session (BlockId) level
    idx = np.arange(len(enc))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=y_session)

    def gather(indices):
        return [enc[i] for i in indices], y_session[indices], [block_ids[i] for i in indices]

    train_seqs, y_train_session, train_bids = gather(train_idx)
    test_seqs, y_test_session, test_bids = gather(test_idx)

    # Build next-event windows from TRAIN sessions only
    train_windows = []
    for s in train_seqs:
        train_windows.extend(make_next_event_windows(s, args.window))

    if len(train_windows) == 0:
        raise RuntimeError("No training windows produced. Try a smaller --window.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepLogLSTM(vocab_size=len(vocab), emb_dim=args.emb_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    loader = DataLoader(NextEventDataset(train_windows), batch_size=args.batch_size, shuffle=True, drop_last=False)

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

    # Save
    meta = {
        "window": args.window,
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "vocab": vocab,
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")

    # Evaluate session-level anomaly detection with top-k rule on TEST sessions
    metrics, cm_path = evaluate_session_level(
        model=model,
        device=device,
        test_block_ids=test_bids,
        test_sequences=test_seqs,
        y_true_session=y_test_session,
        window=args.window,
        topk=args.topk,
        out_dir=out_dir,
    )

    print("\nSession-level metrics (top-k next-event):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print(f"Saved confusion matrix plot to: {cm_path}")

if __name__ == "__main__":
    main()
