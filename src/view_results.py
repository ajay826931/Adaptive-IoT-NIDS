import os
import glob
import sys
import pandas as pd
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "uniform_label")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CLASS_FILE_MAP = {
    'edge_iot': 'classes_edge-iiot_dwn10p.txt',
    'edge-iot': 'classes_edge-iiot_dwn10p.txt',
    'iot_nidd': 'classes_iot-nidd_dwn10p.txt',
    'iot-nidd': 'classes_iot-nidd_dwn10p.txt',
    'ton_iot': 'classes_ton-iot_dwn10p.txt',
    'ton-iot': 'classes_ton-iot_dwn10p.txt',
    'sample_iot': 'classes_sample_iot_dwn10p.txt',
    'sample-iot': 'classes_sample_iot_dwn10p.txt'
}

def get_latest_exp_dir(base_dir=RESULTS_DIR):
    if not os.path.exists(base_dir):
        return None
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return dirs[0]

def detect_classes_file(exp_dir, data_dir=DATA_DIR):
    args_files = glob.glob(os.path.join(exp_dir, "args-*.txt"))
    if args_files:
        with open(args_files[0], 'r') as f:
            for line in f:
                if "datasets:" in line:
                    for key, val in CLASS_FILE_MAP.items():
                        if key in line:
                            path = os.path.join(data_dir, val)
                            if os.path.exists(path):
                                return path
    exp_name = os.path.basename(exp_dir).lower()
    for key, val in CLASS_FILE_MAP.items():
        if key in exp_name:
            path = os.path.join(data_dir, val)
            if os.path.exists(path):
                return path
    return os.path.join(data_dir, 'classes_edge-iiot_dwn10p.txt')

def show_results(exp_dir=None):
    if exp_dir and not os.path.isabs(exp_dir):
        # Resolve relative to project root or cwd
        candidates = [os.path.abspath(exp_dir), os.path.join(PROJECT_ROOT, exp_dir), os.path.join(RESULTS_DIR, exp_dir)]
        for c in candidates:
            if os.path.exists(c):
                exp_dir = c
                break

    if not exp_dir or not os.path.exists(exp_dir):
        exp_dir = get_latest_exp_dir()

    if not exp_dir or not os.path.exists(exp_dir):
        print("[!] No experiment results found.")
        return

    exp_basename = os.path.basename(exp_dir)
    print("=" * 65)
    print(f"       EXPERIMENT RESULTS: {exp_basename.upper()}")
    print("=" * 65)
    
    stdout_files = sorted(glob.glob(os.path.join(exp_dir, "stdout-*.txt")), key=os.path.getmtime, reverse=True)
    if stdout_files:
        latest_log = stdout_files[0]
        print(f"\n[+] Log File: {os.path.basename(latest_log)}")
        with open(latest_log, 'r') as f:
            for line in f:
                if ">>> Test on task" in line or "TAw Acc" in line or "TAg Acc" in line or "Avg.:" in line:
                    print("   ", line.strip())

    metrics_files = sorted(glob.glob(os.path.join(exp_dir, "results/*_per_class_metrics.parquet")), key=os.path.getmtime, reverse=True)
    classes_path = detect_classes_file(exp_dir)
    
    if metrics_files and os.path.exists(classes_path):
        latest_metrics = metrics_files[0]
        classes = open(classes_path).read().splitlines()
        df_metric = pd.read_parquet(latest_metrics)
        
        f1_scores = df_metric['f1_score'].iloc[0]
        recalls = df_metric['recall_score'].iloc[0]
        
        table = []
        for i, cls_name in enumerate(classes[:len(f1_scores)]):
            table.append({
                'Attack / Traffic Class': cls_name,
                'F1-Score (%)': f"{f1_scores[i] * 100:.2f}%",
                'Recall (%)': f"{recalls[i] * 100:.2f}%"
            })
        
        df_table = pd.DataFrame(table)
        print(f"\n[+] Per-Class Breakdown ({os.path.basename(classes_path)}):")
        print("-" * 65)
        print(df_table.to_string(index=False))
        print("-" * 65)
    elif not metrics_files:
        print("\n[i] Run option [3] Compute Metrics to generate per-class breakdown.")

if __name__ == "__main__":
    target_exp = sys.argv[1] if len(sys.argv) > 1 else None
    show_results(target_exp)
