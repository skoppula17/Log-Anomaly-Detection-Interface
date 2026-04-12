from __future__ import annotations
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from src.data.hdfs import make_next_event_windows


def build_freq_model(train_sequences: List[List[int]], window: int) -> Tuple[Dict[int, List[Tuple[int, int]]], Counter]:
    last_to_next: Dict[int, Counter] = {}
    global_next: Counter = Counter()
    for seq in train_sequences:
        for xw, y in make_next_event_windows(seq, window):
            last = xw[-1]
            last_to_next.setdefault(last, Counter())[y] += 1
            global_next[y] += 1
    top_by_last: Dict[int, List[Tuple[int, int]]] = {}
    for last, cnt in last_to_next.items():
        top_by_last[last] = cnt.most_common()
    global_top = global_next.most_common()
    return top_by_last, global_top


def predict_topk(
    top_by_last: Dict[int, List[Tuple[int, int]]],
    global_top: List[Tuple[int, int]],
    window_tokens: List[int],
    topk: int,
) -> List[int]:
    last = window_tokens[-1]
    if last in top_by_last:
        candidates = top_by_last[last]
    else:
        candidates = global_top
    return [e for e, _ in candidates[:topk]]


def session_is_anomalous_freq(
    top_by_last: Dict[int, List[Tuple[int, int]]],
    global_top: List[Tuple[int, int]],
    seq: List[int],
    window: int,
    topk: int,
) -> int:
    windows = make_next_event_windows(seq, window)
    if not windows:
        return 0
    for xw, y_true in windows:
        preds = predict_topk(top_by_last, global_top, xw, topk=topk)
        if y_true not in preds:
            return 1
    return 0
