import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os

def plot_topk_sweep(sweep_csv: Path, out_dir: Path):
    if not sweep_csv.exists():
        print(f"Warning: {sweep_csv} not found.")
        return
        
    df = pd.read_csv(sweep_csv)
    plt.figure(figsize=(8, 6))
    
    # Plot each metric
    plt.plot(df['topk'], df['precision'], marker='o', linewidth=2, label='Precision')
    plt.plot(df['topk'], df['recall'], marker='s', linewidth=2, label='Recall')
    plt.plot(df['topk'], df['f1'], marker='^', linewidth=2, label='F1-Score')
    
    plt.title('Validation Metrics vs Top-K Threshold', fontsize=14, pad=15)
    plt.xlabel('Top-K', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xticks(df['topk'])
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    out_path = out_dir / 'val_sweep_metrics.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_model_comparison(lstm_json: Path, baseline_json: Path, out_dir: Path):
    if not lstm_json.exists() or not baseline_json.exists():
        print("Warning: Metric JSON files not found.")
        return
        
    with open(lstm_json, 'r') as f:
        lstm = json.load(f)
    with open(baseline_json, 'r') as f:
        base = json.load(f)
        
    metrics = ['precision', 'recall', 'f1']
    lstm_vals = [lstm.get(m, 0.0) for m in metrics]
    base_vals = [base.get(m, 0.0) for m in metrics]
    
    x = range(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw bars
    rects1 = ax.bar([i - width/2 for i in x], lstm_vals, width, label='DeepLog LSTM', color='#2b5b84')
    rects2 = ax.bar([i + width/2 for i in x], base_vals, width, label='Frequency Baseline', color='#8b3a3a')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test Set Performance Comparison', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    
    # Add gridlines behind bars
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    out_path = out_dir / 'model_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate plots from run metrics")
    parser.add_argument('--run_dir', type=str, default='runs/cp1', help='Directory containing metrics files')
    parser.add_argument('--out_dir', type=str, default='results/graphs', help='Output directory for generated images')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading metrics from '{run_dir}' and saving plots to '{out_dir}'...")
    
    sweep_csv = run_dir / 'val_sweep.csv'
    plot_topk_sweep(sweep_csv, out_dir)
    
    lstm_json = run_dir / 'metrics.json'
    base_json = run_dir / 'baseline_metrics.json'
    plot_model_comparison(lstm_json, base_json, out_dir)

if __name__ == '__main__':
    main()
