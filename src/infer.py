from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from src.models.deeplog_lstm import DeepLogLSTM
from src.eval import session_is_anomalous
from src.data.hdfs import load_hdfs_event_traces, encode_sequences

def main() -> None:
    ap = argparse.ArgumentParser(description="CLI inference tool to flag anomalous log sessions.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to the trained model point (model.pt)")
    ap.add_argument("--meta", type=str, required=True, help="Path to the training metadata (meta.json)")
    ap.add_argument("--event_traces", type=str, required=True, help="Path to Event_traces.csv to analyze")
    ap.add_argument("--topk", type=int, default=None, help="Top-K threshold for anomaly criteria. Defaults to same used during training if not set. E.g. 9")
    args = ap.parse_args()

    # Load metadata
    meta = json.loads(Path(args.meta).read_text())
    vocab = meta["vocab"]
    window = int(meta["window"])
    emb_dim = int(meta["emb_dim"])
    hidden_dim = int(meta["hidden_dim"])
    topk = args.topk if args.topk is not None else 9 # default 9 or best

    print(f"[INFO] Using window={window}, topk={topk}")

    # Load Data
    print(f"[INFO] Loading target data: {args.event_traces}...")
    block_ids, seqs = load_hdfs_event_traces(args.event_traces)
    print(f"[INFO] Loaded {len(block_ids)} sessions.")

    # Encode Data
    enc = encode_sequences(seqs, vocab)

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepLogLSTM(vocab_size=len(vocab), emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    print("[INFO] Running inference...")
    anomalous_sessions = []
    
    for i, seq in enumerate(enc):
        is_anom = session_is_anomalous(model, device, seq, window=window, topk=topk)
        if is_anom:
            anomalous_sessions.append(block_ids[i])
    
    # Simple Output Report
    total = len(block_ids)
    flagged = len(anomalous_sessions)
    print("=" * 50)
    print("INFERENCE REPORT")
    print("=" * 50)
    print(f"Total sessions reviewed : {total}")
    print(f"Anomalous sessions      : {flagged} ({(flagged/total)*100:.2f}%)")
    print("-" * 50)
    
    if flagged > 0:
        print("Flagged Session IDs (first 20):")
        for bid in anomalous_sessions[:20]:
            print(f" - {bid}")
        if flagged > 20:
            print(f"   ... and {flagged - 20} more.")
    else:
        print("No anomalous sessions detected.")

if __name__ == "__main__":
    main()
