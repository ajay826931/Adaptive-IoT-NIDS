import os
import glob
import pandas as pd
import numpy as np

def show_results(exp_dir="../results/intra_edge-iot_scratch"):
    print("=" * 65)
    print("       ADAPTIVE IoT-NIDS : EXPERIMENT RESULTS SUMMARY        ")
    print("=" * 65)
    
    # 1. Check stdout log for general accuracy
    stdout_files = sorted(glob.glob(os.path.join(exp_dir, "stdout-*.txt")), reverse=True)
    if stdout_files:
        latest_log = stdout_files[0]
        print(f"\n[+] Latest Run Log: {os.path.basename(latest_log)}")
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ">>> Test on task" in line or "TAw Acc" in line or "TAg Acc" in line or "Avg.:" in line:
                    print("   ", line.strip())

    # 2. Check per-class metrics
    metrics_files = sorted(glob.glob(os.path.join(exp_dir, "results/*_per_class_metrics.parquet")), reverse=True)
    classes_path = "../data/uniform_label/classes_edge-iiot_dwn10p.txt"
    
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
        print("\n[+] Per-Class Performance Breakdown:")
        print("-" * 65)
        print(df_table.to_string(index=False))
        print("-" * 65)

if __name__ == "__main__":
    show_results()
