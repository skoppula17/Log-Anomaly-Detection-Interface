# Log Anomaly Detection for Security Analysts - ECE 570 Track 2

This repository implements an end-to-end Track 2 (ProductPrototype) project: This is a SOC tool that learns normal log-event sequences and flags anomalous sessions.
This includes the following features: Deep learning model pipeline, measurable metrics, and consumer facing interface**.

## What this implements (baseline)
- Dataset: LogHub HDFS (preprocessed) files (Event_traces + anomaly labels)
- Model: **DeepLog-style** next-event prediction with an LSTM
- Decision rule: a session is anomalous if any next-event falls outside top-k predictions

## How to Set Up the Project
### 1) Create env + install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2) Download data (LogHub preprocessed)
```bash
python3 -m src.data.download_hdfs --out data/raw
```

This script automatically downloads the real 100k subset from Logpai's Loglizer repository and parses the `HDFS_100k.log_structured.csv` into `Event_traces.csv` and `anomaly_label.csv`.
If LogHub ever changes paths, update URLs inside `src/data/download_hdfs.py`.

### 3) Train baseline
```bash
python3 -m src.train \
  --event_traces data/raw/Event_traces.csv \
  --labels data/raw/anomaly_label.csv \
  --out_dir runs/cp1 \
  --epochs 10
```

### 4) Evaluate (prints metrics + saves confusion matrix)
Training runs evaluation at the end, but you can re-run:
```bash
python3 -m src.eval \
  --ckpt runs/cp1/model.pt \
  --meta runs/cp1/meta.json \
  --event_traces data/raw/Event_traces.csv \
  --labels data/raw/anomaly_label.csv \
  --topk 3
```

### 5) Run Inference (CLI)
You can directly review new/specific sessions via the CLI:
```bash
python3 -m src.infer \
  --ckpt runs/cp1/model.pt \
  --meta runs/cp1/meta.json \
  --event_traces data/raw/Event_traces.csv
```

### 6) Generate Graphs
To automatically generate validation and model comparison charts from your run metrics:
```bash
python3 -m src.plot_metrics
```
Graphs will be saved to `results/graphs/`.

### 7) Run the Analyst Dashboard (Streamlit Frontend)
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
- `src/plot_metrics.py` Visualization generation script
- `sources.md` citations + what this repo is inspired by

## LLM Acknowledgement & Code Origin

In accordance with course policies, the following outlines the code origins for this project:

- **Original Code**: The majority of the repository, including the architecture design, the Streamlit dashboard (`src/app.py`), the core LSTM baseline components, the evaluation logic, and the overall pipeline flow, was authored by me. I also wrote the contents and structure of this `README.md`, but utilized an LLM to assist with formatting it cleanly. 
- **AI-Assisted Code**: Generative AI was used to assist with UI bug fixes, construct graphing/visualization code, and dynamically parse the external HDFS datasets.

Specifically, AI was prompted to write:
1. **HDFS Dataset Parsing** (`src/data/download_hdfs.py: lines 38-47`): Code to parse the raw 100k Loglizer dataset, extracting `BlockId` and generating the event sequences.
   ```python
   # AI-Generated parsing snippet
   df['BlockId'] = df['Content'].str.extract(r'(blk_[-0-9]+)')
   df = df.dropna(subset=['BlockId'])
   traces = df.groupby('BlockId')['EventId'].apply(lambda x: ' '.join(x)).reset_index()
   ```
2. **Graph Generation** (`src/plot_metrics.py: lines 55-56`): Code to visualize the test set performance comparison as a bar chart.
3. **Path Resolution Fix** (`src/app.py: lines 8-9`): Code added to correctly resolve module import paths for the Streamlit dashboard.

## Extra Notes
This repo implements a baseline inspired by the DeepLog approach (next-event prediction + top-k rule) and uses publicly available datasets.

See `sources.md` for references.
