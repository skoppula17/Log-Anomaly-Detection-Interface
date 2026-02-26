import argparse
import os
from pathlib import Path
import requests

# LogHub sometimes stores the preprocessed HDFS artifacts under the HDFS folder.
# If these links break, update them to the correct raw GitHub URLs.
DEFAULT_URLS = {
    "Event_traces.csv": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/Event_traces.csv",
    "anomaly_label.csv": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/anomaly_label.csv",
}

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/raw", help="Output directory for raw files")
    ap.add_argument("--event_traces_url", type=str, default=DEFAULT_URLS["Event_traces.csv"])
    ap.add_argument("--labels_url", type=str, default=DEFAULT_URLS["anomaly_label.csv"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        (args.event_traces_url, out_dir / "Event_traces.csv"),
        (args.labels_url, out_dir / "anomaly_label.csv"),
    ]

    for url, path in targets:
        print(f"Downloading {url} -> {path}")
        _download(url, path)

    print("Done.")

if __name__ == "__main__":
    main()
