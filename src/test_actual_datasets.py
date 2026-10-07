import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import xgboost as xgb
from catboost import CatBoostClassifier

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

DATASET_PATH = r"D:\ajay\jaynak sir\data\Datasets\Edge-IIoTset dataset\Selected dataset for ML and DL\ML-EdgeIIoT-dataset.csv"

# Canonical Attack Taxonomy (Section 2.2 in PROJECT_RULES.md)
CANONICAL_LABEL_MAP = {
    'Normal': 'benign',
    'DDoS_HTTP': 'dos-http',
    'DDoS_ICMP': 'dos-icmp',
    'DDoS_TCP': 'dos-tcp-syn',
    'DDoS_UDP': 'dos-udp',
    'Fingerprinting': 'infog-scan-os',
    'Port_Scanning': 'infog-scan-port',
    'Vulnerability_scanner': 'infog-scan-vuln',
    'Uploading': 'inject-http',
    'SQL_injection': 'inject-sql',
    'XSS': 'inject-xss',
    'Backdoor': 'malwr-backd-tcp',
    'Password': 'malwr-brutef-http',
    'Ransomware': 'malwr-ransomware',
    'MITM': 'mitm-arpspoof'
}

DROP_COLS = [
    'frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
    'http.file_data', 'http.request.full_uri', 'http.request.uri.query', 'http.referer',
    'tcp.payload', 'tcp.options', 'dns.qry.name', 'mqtt.msg', 'mqtt.topic'
]

def load_and_preprocess_balanced_slice(dataset_path=DATASET_PATH, max_per_class=3000, seed=42):
    print(f"\n[Step 1] Loading Balanced Diagnostic Slice from: {os.path.basename(dataset_path)}")
    t0 = time.time()
    
    # Read CSV
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"    * Full file loaded in memory: {len(df):,} total rows, {len(df.columns)} columns ({time.time() - t0:.2f}s)")
    
    # Map to Canonical labels
    df['CANONICAL_LABEL'] = df['Attack_type'].map(CANONICAL_LABEL_MAP).fillna(df['Attack_type'])
    
    # Create Stratified Balanced Sample
    balanced_dfs = []
    for cls_name, grp in df.groupby('CANONICAL_LABEL'):
        n_samples = min(len(grp), max_per_class)
        sampled = grp.sample(n=n_samples, random_state=seed)
        balanced_dfs.append(sampled)
        
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"    * Stratified Balanced Slice extracted: {len(df_balanced):,} flows across {df_balanced['CANONICAL_LABEL'].nunique()} classes")

    # Clean numeric features
    feature_cols = [c for c in df.columns if c not in DROP_COLS and c not in ['Attack_label', 'Attack_type', 'CANONICAL_LABEL']]
    for c in feature_cols:
        df_balanced[c] = pd.to_numeric(df_balanced[c], errors='coerce').fillna(0.0)

    X = df_balanced[feature_cols].values.astype(np.float32)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    unique_classes = sorted(df_balanced['CANONICAL_LABEL'].unique())
    cls_to_id = {c: i for i, c in enumerate(unique_classes)}
    y = df_balanced['CANONICAL_LABEL'].map(cls_to_id).values.astype(int)

    return X, y, unique_classes, feature_cols

def run_diagnostic_evaluation(max_per_class=3000, use_gpu=True, seed=42):
    print("=" * 115)
    print("        STAGE 1: BALANCED DIAGNOSTIC EVALUATION & MODEL LAGGING ANALYSIS        ")
    print(f"        Dataset Source: D:\\ajay\\jaynak sir\\data\\Datasets\\Edge-IIoTset dataset")
    print(f"        Hardware Engine: {'NVIDIA CUDA GPU' if use_gpu else 'CPU'}")
    print("=" * 115)

    # 1. Load balanced data
    X, y, class_names, feature_names = load_and_preprocess_balanced_slice(DATASET_PATH, max_per_class=max_per_class, seed=seed)
    num_classes = len(class_names)

    # 2. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    print(f"\n[Step 2] Data Partitioning:")
    print(f"    * Training Set:   {len(X_train):,} flows")
    print(f"    * Evaluation Set: {len(X_test):,} flows")
    print(f"    * Feature Space:  {X_train.shape[1]} network traffic features")

    # 3. Train Hybrid Dual-Tree Architecture on GPU
    print(f"\n[Step 3] Training Hybrid Dual-Tree Engine (XGBoost + CatBoost)...")
    xgb_device = 'cuda' if use_gpu else 'cpu'
    
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device=xgb_device,
        eval_metric='mlogloss',
        random_state=seed
    )

    cb_task_type = 'GPU' if use_gpu else 'CPU'
    cb_clf = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        task_type=cb_task_type,
        verbose=0,
        random_seed=seed
    )

    t0_fit = time.time()
    xgb_clf.fit(X_train, y_train)
    cb_clf.fit(X_train, y_train)
    t_train_duration = time.time() - t0_fit
    print(f"    [✓] Training Completed on GPU in {t_train_duration:.2f} seconds")

    # 4. Predict & Probability Fusion
    print("\n[Step 4] Running Evaluation & Probability Fusion...")
    t0_inf = time.time()
    p_xgb = xgb_clf.predict_proba(X_test)
    p_cat = cb_clf.predict_proba(X_test)
    t_inf_duration = time.time() - t0_inf
    
    # Fused Probability
    p_fused = 0.5 * p_xgb + 0.5 * p_cat
    y_pred = np.argmax(p_fused, axis=1)

    # Latency
    latency_us = (t_inf_duration / len(X_test)) * 1e6
    throughput = len(X_test) / t_inf_duration

    # 5. Overall Metrics
    overall_acc = accuracy_score(y_test, y_pred) * 100.0
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100.0
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100.0
    macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100.0

    try:
        y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))
        macro_auc = roc_auc_score(y_test_bin, p_fused, average='macro', multi_class='ovr') * 100.0
    except Exception:
        macro_auc = 0.0

    # False Positive Rate (Benign misclassified as Malicious)
    benign_idx = class_names.index('benign') if 'benign' in class_names else 0
    fp = np.sum((y_test == benign_idx) & (y_pred != benign_idx))
    tn = np.sum((y_test == benign_idx) & (y_pred == benign_idx))
    fpr = (fp / (fp + tn + 1e-5)) * 100.0

    # 6. Per-Class Diagnostic & Lag Identification (Option C)
    print("\n" + "=" * 125)
    print("                    PER-CLASS ATTACK DIAGNOSTIC BREAKDOWN & LAGGING IDENTIFICATION                    ")
    print("=" * 125)
    header = f"{'Class Label':<22} | {'Test Samples':<12} | {'Precision [%]':<13} | {'Recall [%]':<11} | {'F1-Score [%]':<12} | {'Diagnosis / Health Status':<25}"
    print(header)
    print("-" * 125)

    per_class_data = []
    cm = confusion_matrix(y_test, y_pred)
    
    lagging_classes = []
    excellent_classes = []

    for c_id, c_name in enumerate(class_names):
        c_samples = int(np.sum(y_test == c_id))
        c_prec = precision_score(y_test == c_id, y_pred == c_id, zero_division=0) * 100.0
        c_rec = recall_score(y_test == c_id, y_pred == c_id, zero_division=0) * 100.0
        c_f1 = f1_score(y_test == c_id, y_pred == c_id, zero_division=0) * 100.0

        if c_f1 >= 95.0:
            status = "[EXCELLENT] Flawless Detection"
            excellent_classes.append((c_name, c_f1))
        elif c_f1 >= 85.0:
            status = "[GOOD] High Accuracy"
        elif c_f1 >= 70.0:
            status = "[MODERATE] Minor Confusion"
            lagging_classes.append((c_name, c_f1, "Moderate confusion"))
        else:
            status = "[CRITICAL LAG] Needs Tuning!"
            lagging_classes.append((c_name, c_f1, "High miss rate"))

        print(f"{c_name:<22} | {c_samples:<12} | {c_prec:<13.2f} | {c_rec:<11.2f} | {c_f1:<12.2f} | {status:<25}")

        per_class_data.append({
            'Attack_Class': c_name,
            'Test_Samples': c_samples,
            'Precision': round(c_prec, 2),
            'Recall': round(c_rec, 2),
            'F1_Score': round(c_f1, 2),
            'Status': status
        })

    print("=" * 125)

    # 7. Deep Lagging Analysis (Confusion Highlights)
    print("\n[+] DEEP CONFUSION & LAGGING ROOT-CAUSE ANALYSIS:")
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 10:  # More than 10 confusions
                confused_pairs.append((class_names[i], class_names[j], int(cm[i, j])))
                
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    if confused_pairs:
        print("    * Top Attack Confusion Pairs (True Class -> Predicted As):")
        for true_cls, pred_cls, count in confused_pairs[:6]:
            pct = (count / np.sum(y_test == class_names.index(true_cls))) * 100.0
            print(f"      - {true_cls}  --->  misclassified as  {pred_cls}: {count} flows ({pct:.1f}%)")
    else:
        print("    * Zero major confusions! Model shows near-perfect separation across classes.")

    # False Negatives (Attacks classified as Benign)
    fn_total = 0
    fn_details = []
    for c_id, c_name in enumerate(class_names):
        if c_name != 'benign':
            fn_count = int(cm[c_id, benign_idx])
            if fn_count > 0:
                fn_total += fn_count
                fn_pct = (fn_count / np.sum(y_test == c_id)) * 100.0
                fn_details.append((c_name, fn_count, fn_pct))

    print(f"\n    * Total False Negatives (Attacks missed as Benign): {fn_total} flows")
    fn_details.sort(key=lambda x: x[1], reverse=True)
    for c_name, cnt, pct in fn_details[:5]:
        print(f"      - {c_name}: {cnt} attacks escaped as Benign ({pct:.2f}% miss rate)")

    # 8. Overall Benchmark Summary
    print("\n" + "=" * 70)
    print("OVERALL MODEL BENCHMARK ON ACTUAL DATASET")
    print("=" * 70)
    print(f"  * Macro-F1 Score:               {macro_f1:.2f}%")
    print(f"  * Multiclass Accuracy:          {overall_acc:.2f}%")
    print(f"  * Macro Recall:                 {macro_rec:.2f}%")
    print(f"  * Macro Precision:              {macro_prec:.2f}%")
    print(f"  * Multiclass AUC:               {macro_auc:.2f}%")
    print(f"  * False Positive Rate (FPR):    {fpr:.2f}%")
    print(f"  * Single Flow Latency:          {latency_us:.2f} µs/flow")
    print(f"  * Real-Time Inference Speed:    {throughput:,.0f} flows/sec")
    print("=" * 70)

    # 9. Export Reports
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_out = os.path.join(results_dir, 'actual_dataset_diagnostic_report.csv')
    md_out = os.path.join(results_dir, 'actual_dataset_diagnostic_report.md')

    df_report = pd.DataFrame(per_class_data)
    df_report.to_csv(csv_out, index=False)
    try:
        df_report.to_markdown(md_out, index=False)
    except Exception:
        pass

    print(f"\n[+] Diagnostic report exported successfully:")
    print(f"    - CSV:      {csv_out}")
    print(f"    - Markdown: {md_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 1 Diagnostic Evaluation on Actual Dataset")
    parser.add_argument('--max-per-class', type=int, default=3000, help='Max balanced samples per class')
    parser.add_argument('--cpu', action='store_true', help='Force CPU instead of GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    use_gpu = not args.cpu
    run_diagnostic_evaluation(max_per_class=args.max_per_class, use_gpu=use_gpu, seed=args.seed)
