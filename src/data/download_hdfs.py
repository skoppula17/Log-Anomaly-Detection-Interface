import argparse
import os
from pathlib import Path
import requests
import random
import csv
import pandas as pd
import re

# We now use the real 100k HDFS log subset from Logpai's Loglizer repository
DEFAULT_URLS = {
    "HDFS_100k.log_structured.csv": "https://raw.githubusercontent.com/logpai/loglizer/master/data/HDFS/HDFS_100k.log_structured.csv",
    "anomaly_label.csv": "https://raw.githubusercontent.com/logpai/loglizer/master/data/HDFS/anomaly_label.csv",
}

def generate_synthetic_data(out_dir: Path, num_blocks: int = 2000, anomaly_ratio: float = 0.05):
    print("Generating synthetic HDFS dataset for testing...")
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "Event_traces.csv"
    labels_path = out_dir / "anomaly_label.csv"
    
    with open(traces_path, "w", newline="") as ft, open(labels_path, "w", newline="") as fl:
        tw = csv.writer(ft)
        lw = csv.writer(fl)
        tw.writerow(["BlockId", "EventSequence"])
        lw.writerow(["BlockId", "Label"])
        
        for i in range(num_blocks):
            blk = f"blk_{-random.randint(1000000000000000000, 9000000000000000000)}"
            is_anomaly = random.random() < anomaly_ratio
            seq = ["E7", "E13", "E13", "E1", "E1", "E11", "E11", "E10", "E14"]
            if random.random() < 0.5:
                seq.append("E14")
            
            if is_anomaly:
                seq.insert(random.randint(1, len(seq)-1), random.choice(["E3", "E4", "E8"]))
                lw.writerow([blk, "Anomaly"])
            else:
                lw.writerow([blk, "Normal"])
                
            tw.writerow([blk, " ".join(seq)])

def process_loglizer_hdfs(log_path: Path, out_path: Path):
    print("Extracting BlockIds and converting raw logs into event sequences...")
    df = pd.read_csv(log_path)
    
    # Extract the HDFS BlockId from the Content column
    df['BlockId'] = df['Content'].str.extract(r'(blk_[-0-9]+)')
    
    # Drop rows that don't belong to any block
    df = df.dropna(subset=['BlockId'])
    
    # Group the EventIds sequentially by BlockId
    traces = df.groupby('BlockId')['EventId'].apply(lambda x: ' '.join(x)).reset_index()
    traces.rename(columns={'EventId': 'EventSequence'}, inplace=True)
    
    traces.to_csv(out_path, index=False)
    print(f"Successfully generated {len(traces)} authentic event sequences from the 100k raw logs.")

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/raw", help="Output directory for raw files")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_structured_path = out_dir / "HDFS_100k.log_structured.csv"
    labels_path = out_dir / "anomaly_label.csv"
    traces_path = out_dir / "Event_traces.csv"

    try:
        # Download the real 100k subset and real labels from Loglizer
        _download(DEFAULT_URLS["HDFS_100k.log_structured.csv"], log_structured_path)
        _download(DEFAULT_URLS["anomaly_label.csv"], labels_path)
        
        # Process the raw 100k log lines into Event_traces.csv
        process_loglizer_hdfs(log_structured_path, traces_path)
        
    except Exception as e:
        print(f"Download or processing failed: {e}")
        generate_synthetic_data(out_dir)

    print("Done.")

if __name__ == "__main__":
    main()
