# Log Anomaly Detection (HDFS) - Checkpoint 1 Starter Repo

This repo is a **checkpoint-1-ready** baseline for Track 2 (ProductPrototype): a SOC-style tool that learns normal
log-event sequences and flags anomalous sessions.

Scope for checkpoint 1: **model training + a measurable metric** (session-level Precision/Recall/F1).

## What this implements (baseline)
- Dataset: LogHub HDFS *preprocessed* files (Event_traces + anomaly labels)
- Model: **DeepLog-style** next-event prediction with an LSTM
- Decision rule: a session is anomalous if any next-event falls outside top-k predictions

## Quickstart
### 1) Create env + install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download data (LogHub preprocessed)
```bash
python -m src.data.download_hdfs --out data/raw
```

This script downloads:
- `Event_traces.csv` (BlockId -> event sequence)
- `anomaly_label.csv` (BlockId -> label)

If LogHub ever changes paths, update URLs inside `src/data/download_hdfs.py`.

### 3) Train baseline
```bash
python -m src.train   --event_traces data/raw/Event_traces.csv   --labels data/raw/anomaly_label.csv   --out_dir runs/cp1   --epochs 2   --topk 9
```

### 4) Evaluate (prints metrics + saves confusion matrix)
Training runs evaluation at the end, but you can re-run:
```bash
python -m src.eval   --ckpt runs/cp1/model.pt   --meta runs/cp1/meta.json   --event_traces data/raw/Event_traces.csv   --labels data/raw/anomaly_label.csv   --topk 9
```

## Repo layout
- `src/data/` download + parsing utilities
- `src/models/` LSTM baseline
- `src/train.py` training loop + end-of-run evaluation
- `src/eval.py` standalone evaluation
- `sources.md` citations + what this repo is inspired by

## Notes on attribution
This repo **does not copy** code from external repositories. It implements a baseline inspired by
the DeepLog approach (next-event prediction + top-k rule) and uses publicly available datasets.

See `sources.md` for references.
