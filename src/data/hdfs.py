from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class HDFSData:
    block_ids: List[str]
    sequences: List[List[str]]   # event IDs as strings (e.g., "E1", "E22")
    labels: np.ndarray           # 0 normal, 1 anomaly

def load_hdfs_event_traces(event_traces_csv: str) -> Tuple[List[str], List[List[str]]]:
    df = pd.read_csv(event_traces_csv)
    # Common column names in LogHub preprocessing
    # Expect something like: BlockId, EventSequence
    if "BlockId" not in df.columns:
        # try case variations
        for c in df.columns:
            if c.lower() == "blockid":
                df = df.rename(columns={c: "BlockId"})
                break
    if "EventSequence" not in df.columns:
        # try common alternatives
        for c in df.columns:
            if c.lower() in ("eventsequence", "event_sequence", "events"):
                df = df.rename(columns={c: "EventSequence"})
                break

    if "BlockId" not in df.columns or "EventSequence" not in df.columns:
        raise ValueError(f"Unexpected columns in {event_traces_csv}: {list(df.columns)}")

    block_ids = df["BlockId"].astype(str).tolist()
    seqs = []
    for s in df["EventSequence"].astype(str).tolist():
        # EventSequence is usually space-separated like: "E1 E2 E3"
        tokens = [t.strip() for t in s.split() if t.strip()]
        seqs.append(tokens)
    return block_ids, seqs

def load_hdfs_labels(labels_csv: str) -> Dict[str, int]:
    df = pd.read_csv(labels_csv)
    # Expect something like: BlockId, Label
    if "BlockId" not in df.columns:
        for c in df.columns:
            if c.lower() == "blockid":
                df = df.rename(columns={c: "BlockId"})
                break
    if "Label" not in df.columns:
        for c in df.columns:
            if c.lower() in ("label", "anomaly", "is_anomaly"):
                df = df.rename(columns={c: "Label"})
                break
    if "BlockId" not in df.columns or "Label" not in df.columns:
        raise ValueError(f"Unexpected columns in {labels_csv}: {list(df.columns)}")

    # labels usually "Normal"/"Anomaly" or 0/1
    out: Dict[str, int] = {}
    for bid, lab in zip(df["BlockId"].astype(str), df["Label"]):
        if isinstance(lab, str):
            v = 1 if lab.strip().lower() in ("anomaly", "abnormal", "1", "true") else 0
        else:
            v = 1 if int(lab) == 1 else 0
        out[bid] = v
    return out

def build_vocab(seqs: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in seqs:
        for e in s:
            counts[e] = counts.get(e, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for e, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            vocab[e] = len(vocab)
    return vocab

def encode_sequences(seqs: List[List[str]], vocab: Dict[str, int]) -> List[List[int]]:
    unk = vocab["<UNK>"]
    return [[vocab.get(e, unk) for e in s] for s in seqs]

def align_labels(block_ids: List[str], labels_map: Dict[str, int]) -> np.ndarray:
    y = np.zeros(len(block_ids), dtype=np.int64)
    for i, bid in enumerate(block_ids):
        y[i] = int(labels_map.get(bid, 0))
    return y

def make_next_event_windows(seq: List[int], window: int) -> List[Tuple[List[int], int]]:
    # returns list of (context_window, next_event)
    if len(seq) <= window:
        return []
    out = []
    for i in range(len(seq) - window):
        x = seq[i:i+window]
        y = seq[i+window]
        out.append((x, y))
    return out
