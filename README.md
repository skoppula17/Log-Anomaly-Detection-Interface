# Log Anomaly Detection for Security Analysts - ECE 570 Track 2

This repository implements an end-to-end Track 2 (ProductPrototype) project: a SOC-style tool that learns normal log-event sequences and flags anomalous sessions.

Scope: **Deep learning model pipeline + measurable metrics + analyst-facing tool**.

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

### 5) Run Inference (CLI)
You can directly review new/specific sessions via the CLI:
```bash
python -m src.infer \
  --ckpt runs/cp1/model.pt \
  --meta runs/cp1/meta.json \
  --event_traces data/raw/Event_traces.csv
```

### 6) Run the Analyst Dashboard (Streamlit Frontend)
To provide a true Product Prototype experience for a security analyst, there is a simple web interface:
```bash
streamlit run src/app.py
```
Upload your `Event_traces.csv` to see a detailed breakdown of anomalous vs normal sessions!

## Repo layout
- `src/data/` download + parsing utilities
- `src/models/` LSTM model & Frequency baseline
- `src/train.py` training loop + end-of-run evaluation
- `src/eval.py` standalone evaluation against baseline
- `src/infer.py` CLI user tool
- `src/app.py` Streamlit analyst dashboard prototype
- `sources.md` citations + what this repo is inspired by

## Notes on attribution
This repo **does not copy** code from external repositories. It implements a baseline inspired by
the DeepLog approach (next-event prediction + top-k rule) and uses publicly available datasets.

See `sources.md` for references.
