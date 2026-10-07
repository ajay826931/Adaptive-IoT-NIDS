import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

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

CLASS_FILE_MAP = {
    'edge_iot': 'classes_edge-iiot_dwn10p.txt',
    'edge-iot': 'classes_edge-iiot_dwn10p.txt',
    'edge-iiot': 'classes_edge-iiot_dwn10p.txt',
    'iot_nidd': 'classes_iot-nidd_dwn10p.txt',
    'iot-nidd': 'classes_iot-nidd_dwn10p.txt',
    'ton_iot': 'classes_ton-iot_dwn10p.txt',
    'ton-iot': 'classes_ton-iot_dwn10p.txt',
}

DISPLAY_NAMES = {
    'edge_iot': 'Edge-IIoT',
    'iot_nidd': 'IoT-NID',
    'ton_iot': 'TON_IoT'
}

def load_class_names(dataset_name):
    fname = CLASS_FILE_MAP.get(dataset_name, f"classes_{dataset_name}_dwn10p.txt")
    classes_path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', fname)
    if not os.path.exists(classes_path):
        classes_path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', f"classes_{dataset_name.replace('_', '-')}_dwn10p.txt")
    return [line.strip() for line in open(classes_path).readlines() if line.strip()]

DATASET_FILE_MAP = {
    'edge_iot': 'edge-iiot_dwn10p_prep1.parquet',
    'edge-iot': 'edge-iiot_dwn10p_prep1.parquet',
    'edge-iiot': 'edge-iiot_dwn10p_prep1.parquet',
    'edge_iiot': 'edge-iiot_dwn10p_prep1.parquet',
    'ton_iot': 'ton-iot_dwn10p_prep1.parquet',
    'ton-iot': 'ton-iot_dwn10p_prep1.parquet',
    'iot_nidd': 'iot-nidd_dwn10p_prep1.parquet',
    'iot-nidd': 'iot-nidd_dwn10p_prep1.parquet',
    'iot_nid': 'iot-nidd_dwn10p_prep1.parquet',
    'iot-nid': 'iot-nidd_dwn10p_prep1.parquet',
}

def load_parquet_dataset(dataset_name):
    fname = DATASET_FILE_MAP.get(dataset_name, f"{dataset_name}_dwn10p_prep1.parquet")
    path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', fname)
    if not os.path.exists(path):
        fname_alt = f"{dataset_name.replace('_', '-')}_dwn10p_prep1.parquet"
        path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', fname_alt)
    df = pd.read_parquet(path)
    return df

def extract_multidimensional_features(df, num_pkts=10):
    """
    Section 8 Taxonomy: Multi-Dimensional Feature Engineering Engine
    Transforms [10 x 4] biflow matrices (PL, IAT, DIR, WIN) into a 77-dimensional feature space:
    1. Statistical Dimensions (Mean, Std, Min, Max, Median, Var, Sum)
    2. Temporal Dynamics (Duration, Packet Rate, Byte Rate, IAT Jitter, Jitter Std)
    3. Directional Flow (Fwd/Bwd counts, Direction Ratio, F/B Bytes, Direction Transition Frequency)
    4. Flow Volume & Throughput (Total Bytes, Mean Packet Size, Packet Ratio, Window Dynamic)
    5. Burst Dynamics (Burst count, Max burst, Mean burst size)
    6. Raw sequence values (10x4 = 40 features)
    """
    pl = np.vstack(df['SCALED_PL'].values)[:, :num_pkts]        # [N, 10]
    iat = np.vstack(df['SCALED_IAT'].values)[:, :num_pkts]      # [N, 10]
    direction = np.vstack(df['SCALED_DIR'].values)[:, :num_pkts]  # [N, 10]
    win = np.vstack(df['SCALED_WIN'].values)[:, :num_pkts]      # [N, 10]

    feats = []

    # 1. Statistical Dimensions for PL, IAT, WIN
    for arr in [pl, iat, win]:
        feats.append(np.mean(arr, axis=1, keepdims=True))
        feats.append(np.std(arr, axis=1, keepdims=True))
        feats.append(np.min(arr, axis=1, keepdims=True))
        feats.append(np.max(arr, axis=1, keepdims=True))
        feats.append(np.median(arr, axis=1, keepdims=True))
        feats.append(np.var(arr, axis=1, keepdims=True))
        feats.append(np.sum(arr, axis=1, keepdims=True))

    # 2. Temporal Dynamics
    duration = np.sum(iat, axis=1, keepdims=True)
    pkt_rate = float(num_pkts) / (duration + 1e-5)
    tot_bytes = np.sum(pl, axis=1, keepdims=True)
    byte_rate = tot_bytes / (duration + 1e-5)
    iat_diff = np.abs(np.diff(iat, axis=1))
    iat_jitter = np.mean(iat_diff, axis=1, keepdims=True)
    iat_jitter_std = np.std(iat_diff, axis=1, keepdims=True)
    feats.extend([duration, pkt_rate, byte_rate, iat_jitter, iat_jitter_std])

    # 3. Directional Flow Characteristics
    fwd_pkts = np.sum(direction > 0, axis=1, keepdims=True)
    bwd_pkts = np.sum(direction <= 0, axis=1, keepdims=True)
    dir_ratio = fwd_pkts / (bwd_pkts + 1e-5)
    fwd_bytes = np.sum(pl * (direction > 0), axis=1, keepdims=True)
    bwd_bytes = np.sum(pl * (direction <= 0), axis=1, keepdims=True)
    bytes_ratio = fwd_bytes / (bwd_bytes + 1e-5)
    dir_transitions = np.sum(np.abs(np.diff(direction, axis=1)) > 0, axis=1, keepdims=True)
    feats.extend([fwd_pkts, bwd_pkts, dir_ratio, fwd_bytes, bwd_bytes, bytes_ratio, dir_transitions])

    # 4. Flow Volume & Throughput
    avg_pkt_sz = tot_bytes / float(num_pkts)
    pl_max_min = (np.max(pl, axis=1, keepdims=True) + 1e-5) / (np.min(pl, axis=1, keepdims=True) + 1e-5)
    win_max_min = (np.max(win, axis=1, keepdims=True) + 1e-5) / (np.min(win, axis=1, keepdims=True) + 1e-5)
    feats.extend([avg_pkt_sz, pl_max_min, win_max_min])

    # 5. Burst Dynamics (packets with IAT < median IAT)
    median_iat = np.median(iat, axis=1, keepdims=True)
    is_burst = (iat < median_iat).astype(float)
    burst_count = np.sum(is_burst, axis=1, keepdims=True)
    feats.append(burst_count)

    # 6. Raw Flattened Values (40 features)
    feats.extend([pl, iat, direction, win])

    X = np.hstack(feats)
    y = df['ENC_LABEL'].values.astype(int)
    is_train = df['IS_TRAIN'].values.astype(bool)

    return X, y, is_train

class HybridDualTreeNIDS:
    """
    Section 7 Architecture: Hybrid Ensemble NIDS
    - XGBoost Classifier (P_XGB)
    - CatBoost Classifier (P_CAT)
    - Soft Probability Fusion: P_Ensemble = alpha * P_XGB + (1 - alpha) * P_CAT
    - Source Memory Replay Buffer for continual domain adaptation
    """
    def __init__(self, num_classes, alpha=0.5, use_gpu=True):
        self.num_classes = num_classes
        self.alpha = alpha
        self.use_gpu = use_gpu

        xgb_device = 'cuda' if use_gpu else 'cpu'

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device=xgb_device,
            eval_metric='mlogloss',
            random_state=42
        )

        cb_task_type = 'GPU' if use_gpu else 'CPU'
        self.cat_model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            task_type=cb_task_type,
            verbose=0,
            random_seed=42
        )

        self.seen_classes = None
        self.class_to_local = None
        self.memory_X = None
        self.memory_y = None

    def _prepare_labels(self, y):
        self.seen_classes = np.sort(np.unique(y))
        self.class_to_local = {c: i for i, c in enumerate(self.seen_classes)}
        y_local = np.array([self.class_to_local[c] for c in y])
        return y_local

    def fit(self, X, y):
        y_local = self._prepare_labels(y)
        self.xgb_model.fit(X, y_local)
        self.cat_model.fit(X, y_local)

    def predict_proba(self, X):
        p_xgb_local = self.xgb_model.predict_proba(X)
        p_cat_local = self.cat_model.predict_proba(X)

        n_samples = X.shape[0]
        p_xgb_global = np.zeros((n_samples, self.num_classes), dtype=float)
        p_cat_global = np.zeros((n_samples, self.num_classes), dtype=float)

        for local_idx, global_cls in enumerate(self.seen_classes):
            if local_idx < p_xgb_local.shape[1]:
                p_xgb_global[:, global_cls] = p_xgb_local[:, local_idx]
            if local_idx < p_cat_local.shape[1]:
                p_cat_global[:, global_cls] = p_cat_local[:, local_idx]

        p_fused = self.alpha * p_xgb_global + (1.0 - self.alpha) * p_cat_global
        return p_fused, p_xgb_global, p_cat_global

    def predict(self, X):
        p_fused, _, _ = self.predict_proba(X)
        return np.argmax(p_fused, axis=1)

    def store_exemplars(self, X, y, exemplars_per_class=50):
        """Build Source Memory Replay Buffer"""
        mem_x_list = []
        mem_y_list = []
        unique_classes = np.unique(y)
        for c in unique_classes:
            idx = np.where(y == c)[0]
            if len(idx) > exemplars_per_class:
                selected = np.random.choice(idx, size=exemplars_per_class, replace=False)
            else:
                selected = idx
            mem_x_list.append(X[selected])
            mem_y_list.append(y[selected])
        self.memory_X = np.vstack(mem_x_list)
        self.memory_y = np.concatenate(mem_y_list)

    def incremental_update(self, X_tgt, y_tgt):
        """Update models on joint Target Batch + Memory Buffer"""
        if self.memory_X is not None and len(self.memory_X) > 0:
            X_joint = np.vstack([self.memory_X, X_tgt])
            y_joint = np.concatenate([self.memory_y, y_tgt])
        else:
            X_joint = X_tgt
            y_joint = y_tgt

        y_local = self._prepare_labels(y_joint)
        self.xgb_model.fit(X_joint, y_local)
        self.cat_model.fit(X_joint, y_local)

def evaluate_predictions(y_true, y_pred, y_prob, num_classes):
    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100.0

    # Macro AUC
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1]) * 100.0
        else:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr') * 100.0
    except Exception:
        auc = 0.0

    # False Positive Rate (FPR): Benign (class 0) misclassified as attack
    benign_mask = (y_true == 0)
    if np.sum(benign_mask) > 0:
        fp = np.sum((y_true == 0) & (y_pred != 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fpr = (fp / (fp + tn + 1e-5)) * 100.0
    else:
        fpr = 0.0

    return {
        'acc': acc,
        'f1': macro_f1,
        'rec': macro_rec,
        'prec': macro_prec,
        'auc': auc,
        'fpr': fpr
    }

def run_hybrid_ensemble_experiment(src_name='edge_iot', tgt_name='ton_iot', use_gpu=True):
    print("=" * 110)
    print("     TASK 5: HYBRID ENSEMBLE ARCHITECTURE & DUAL-TREE ENGINE (XGBoost + CatBoost)     ")
    print(f"     Source Domain: {DISPLAY_NAMES.get(src_name, src_name)}  -->  Target Domain: {DISPLAY_NAMES.get(tgt_name, tgt_name)}")
    print(f"     Engine Acceleration: {'NVIDIA CUDA GPU' if use_gpu else 'CPU'}")
    print("=" * 110)

    # 1. Load Data
    t0 = time.time()
    print("\n[Step 1] Loading Parquet datasets...")
    df_src = load_parquet_dataset(src_name)
    df_tgt = load_parquet_dataset(tgt_name)
    src_classes = load_class_names(src_name)
    tgt_classes = load_class_names(tgt_name)
    print(f"    * Source ({DISPLAY_NAMES.get(src_name, src_name)}): {len(df_src)} biflows, {len(src_classes)} classes")
    print(f"    * Target ({DISPLAY_NAMES.get(tgt_name, tgt_name)}): {len(df_tgt)} biflows, {len(tgt_classes)} classes")

    # Unified label space
    all_classes = sorted(list(set(src_classes).union(set(tgt_classes))))
    class_to_global = {c: i for i, c in enumerate(all_classes)}
    num_global_classes = len(all_classes)
    src_to_global = {i: class_to_global[c] for i, c in enumerate(src_classes)}
    tgt_to_global = {i: class_to_global[c] for i, c in enumerate(tgt_classes)}

    # 2. Multi-Dimensional Feature Engineering
    print("\n[Step 2] Executing Multi-Dimensional Feature Engineering Engine (Section 8)...")
    X_src_all, y_src_raw, is_trn_src = extract_multidimensional_features(df_src)
    X_tgt_all, y_tgt_raw, is_trn_tgt = extract_multidimensional_features(df_tgt)
    y_src_global = np.array([src_to_global[k] for k in y_src_raw])
    y_tgt_global = np.array([tgt_to_global[k] for k in y_tgt_raw])

    X_src_trn, y_src_trn = X_src_all[is_trn_src], y_src_global[is_trn_src]
    X_src_tst, y_src_tst = X_src_all[~is_trn_src], y_src_global[~is_trn_src]

    X_tgt_trn, y_tgt_trn = X_tgt_all[is_trn_tgt], y_tgt_global[is_trn_tgt]
    X_tgt_tst, y_tgt_tst = X_tgt_all[~is_trn_tgt], y_tgt_global[~is_trn_tgt]

    print(f"    [+] Extracted {X_src_trn.shape[1]} Engineered Features per Flow:")
    print("        - Statistical (Mean, Std, Min, Max, Median, Var, Sum)")
    print("        - Temporal Dynamics (Duration, Packet Rate, Byte Rate, Jitter)")
    print("        - Directional Characteristics (F/B Packets, Ratios, Transitions)")
    print("        - Flow Volume & Burst Dynamics")
    print("        - Raw Flow Matrix Flattening")
    print(f"    [+] Source Train: {len(X_src_trn)} | Test: {len(X_src_tst)}")
    print(f"    [+] Target Train: {len(X_tgt_trn)} | Test: {len(X_tgt_tst)} ({time.time() - t0:.2f}s)")

    # 3. Train Initial NIDS on Source Domain
    print("\n[Step 3] Training Initial Dual-Tree NIDS on Source Domain...")
    nids = HybridDualTreeNIDS(num_classes=num_global_classes, alpha=0.5, use_gpu=use_gpu)

    t_fit_start = time.time()
    nids.fit(X_src_trn, y_src_trn)
    fit_duration = time.time() - t_fit_start
    print(f"    [+] Initial Training Completed in {fit_duration:.2f}s (XGBoost + CatBoost)")

    # 4. Evaluate Initial Models on Source Test
    p_fused, p_xgb, p_cat = nids.predict_proba(X_src_tst)
    y_pred_fused = np.argmax(p_fused, axis=1)
    y_pred_xgb = np.argmax(p_xgb, axis=1)
    y_pred_cat = np.argmax(p_cat, axis=1)

    eval_initial_fused = evaluate_predictions(y_src_tst, y_pred_fused, p_fused, num_global_classes)
    eval_initial_xgb = evaluate_predictions(y_src_tst, y_pred_xgb, p_xgb, num_global_classes)
    eval_initial_cat = evaluate_predictions(y_src_tst, y_pred_cat, p_cat, num_global_classes)

    print("\n[Step 4] Baseline Source Domain Performance:")
    print(f"    * Standalone XGBoost:  F1 = {eval_initial_xgb['f1']:.2f}% | Acc = {eval_initial_xgb['acc']:.2f}% | AUC = {eval_initial_xgb['auc']:.2f}% | FPR = {eval_initial_xgb['fpr']:.2f}%")
    print(f"    * Standalone CatBoost: F1 = {eval_initial_cat['f1']:.2f}% | Acc = {eval_initial_cat['acc']:.2f}% | AUC = {eval_initial_cat['auc']:.2f}% | FPR = {eval_initial_cat['fpr']:.2f}%")
    print(f"    * HYBRID ENSEMBLE:     F1 = {eval_initial_fused['f1']:.2f}% | Acc = {eval_initial_fused['acc']:.2f}% | AUC = {eval_initial_fused['auc']:.2f}% | FPR = {eval_initial_fused['fpr']:.2f}%")

    # 5. Incremental Continual Adaptation with Memory Replay Buffer
    print("\n[Step 5] Building Source Memory Replay Buffer & Adapting to Target...")
    nids.store_exemplars(X_src_trn, y_src_trn, exemplars_per_class=40)
    print(f"    [+] Stored {len(nids.memory_X)} Source Exemplars in Replay Buffer")

    t_inc_start = time.time()
    nids.incremental_update(X_tgt_trn, y_tgt_trn)
    inc_duration = time.time() - t_inc_start
    print(f"    [+] Incremental Adaptation Completed in {inc_duration:.2f}s")

    # 6. Multi-Dimensional Evaluation on Both Domains
    print("\n[Step 6] Comprehensive Evaluation of Updated Hybrid Ensemble...")
    # Joint test set
    X_joint_tst = np.vstack([X_src_tst, X_tgt_tst])
    y_joint_tst = np.concatenate([y_src_tst, y_tgt_tst])

    t_inf_start = time.time()
    p_fused_joint, p_xgb_joint, p_cat_joint = nids.predict_proba(X_joint_tst)
    t_inf_total = time.time() - t_inf_start
    latency_us_per_flow = (t_inf_total / len(X_joint_tst)) * 1e6
    throughput_flows_sec = len(X_joint_tst) / t_inf_total

    y_pred_joint_fused = np.argmax(p_fused_joint, axis=1)
    y_pred_joint_xgb = np.argmax(p_xgb_joint, axis=1)
    y_pred_joint_cat = np.argmax(p_cat_joint, axis=1)

    eval_fused_joint = evaluate_predictions(y_joint_tst, y_pred_joint_fused, p_fused_joint, num_global_classes)
    eval_xgb_joint = evaluate_predictions(y_joint_tst, y_pred_joint_xgb, p_xgb_joint, num_global_classes)
    eval_cat_joint = evaluate_predictions(y_joint_tst, y_pred_joint_cat, p_cat_joint, num_global_classes)

    # Class subsets: Old (Source-only), New (Target-only), Common (Shared)
    src_set = set(src_to_global.values())
    tgt_set = set(tgt_to_global.values())
    old_classes = list(src_set - tgt_set)
    new_classes = list(tgt_set - src_set)
    com_classes = list(src_set.intersection(tgt_set))

    # Evaluate subsets on joint test
    per_class_f1 = f1_score(y_joint_tst, y_pred_joint_fused, average=None, zero_division=0) * 100.0
    f1_old = np.mean([per_class_f1[c] for c in old_classes]) if old_classes else 0.0
    f1_new = np.mean([per_class_f1[c] for c in new_classes]) if new_classes else 0.0
    f1_com = np.mean([per_class_f1[c] for c in com_classes]) if com_classes else 0.0

    # 7. Print Master Summary Table
    print("\n" + "=" * 115)
    print("MASTER ABLATION & HYBRID ENSEMBLE BENCHMARK TABLE")
    print("=" * 115)
    header = f"{'Model Architecture':<28} | {'Macro-F1 [%]':<13} | {'Accuracy [%]':<13} | {'Recall [%]':<11} | {'Precision [%]':<14} | {'AUC [%]':<9} | {'FPR [%]':<8}"
    print(header)
    print("-" * 115)
    print(f"{'Standalone XGBoost':<28} | {eval_xgb_joint['f1']:<13.2f} | {eval_xgb_joint['acc']:<13.2f} | {eval_xgb_joint['rec']:<11.2f} | {eval_xgb_joint['prec']:<14.2f} | {eval_xgb_joint['auc']:<9.2f} | {eval_xgb_joint['fpr']:<8.2f}")
    print(f"{'Standalone CatBoost':<28} | {eval_cat_joint['f1']:<13.2f} | {eval_cat_joint['acc']:<13.2f} | {eval_cat_joint['rec']:<11.2f} | {eval_cat_joint['prec']:<14.2f} | {eval_cat_joint['auc']:<9.2f} | {eval_cat_joint['fpr']:<8.2f}")
    print(f"{'HYBRID ENSEMBLE (P_XGB + P_CAT)':<28} | {eval_fused_joint['f1']:<13.2f} | {eval_fused_joint['acc']:<13.2f} | {eval_fused_joint['rec']:<11.2f} | {eval_fused_joint['prec']:<14.2f} | {eval_fused_joint['auc']:<9.2f} | {eval_fused_joint['fpr']:<8.2f}")
    print("=" * 115)

    print("\n[+] CLASS PARTITION BREAKDOWN (Continual Adaptation Metric):")
    print(f"    - Old Classes (Source-Only Retained): F1 = {f1_old:.2f}%")
    print(f"    - Common Classes (Shared across S & T): F1 = {f1_com:.2f}%")
    print(f"    - New Classes (Target-Only Learned):   F1 = {f1_new:.2f}%")
    print(f"    - All Classes (Dataset-Agnostic C_all): F1 = {eval_fused_joint['f1']:.2f}%")

    print("\n[+] COMPUTATIONAL COMPLEXITY & REAL-TIME PERFORMANCE:")
    print(f"    - Single Flow Inference Latency: {latency_us_per_flow:.2f} µs/flow")
    print(f"    - Real-Time Throughput:           {throughput_flows_sec:,.0f} flows/second")
    print(f"    - Training Time (Initial + Adapt):{fit_duration + inc_duration:.2f} seconds")

    # 8. Export Results
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, 'hybrid_ensemble_results.csv')
    md_file = os.path.join(results_dir, 'hybrid_ensemble_results.md')

    summary_df = pd.DataFrame([
        {
            'Model': 'Standalone XGBoost',
            'Macro_F1': round(eval_xgb_joint['f1'], 2),
            'Accuracy': round(eval_xgb_joint['acc'], 2),
            'Recall': round(eval_xgb_joint['rec'], 2),
            'Precision': round(eval_xgb_joint['prec'], 2),
            'AUC': round(eval_xgb_joint['auc'], 2),
            'FPR': round(eval_xgb_joint['fpr'], 2),
            'Latency_us': round(latency_us_per_flow, 2),
            'Throughput_fps': round(throughput_flows_sec, 0)
        },
        {
            'Model': 'Standalone CatBoost',
            'Macro_F1': round(eval_cat_joint['f1'], 2),
            'Accuracy': round(eval_cat_joint['acc'], 2),
            'Recall': round(eval_cat_joint['rec'], 2),
            'Precision': round(eval_cat_joint['prec'], 2),
            'AUC': round(eval_cat_joint['auc'], 2),
            'FPR': round(eval_cat_joint['fpr'], 2),
            'Latency_us': round(latency_us_per_flow, 2),
            'Throughput_fps': round(throughput_flows_sec, 0)
        },
        {
            'Model': 'HYBRID ENSEMBLE (Fusion)',
            'Macro_F1': round(eval_fused_joint['f1'], 2),
            'Accuracy': round(eval_fused_joint['acc'], 2),
            'Recall': round(eval_fused_joint['rec'], 2),
            'Precision': round(eval_fused_joint['prec'], 2),
            'AUC': round(eval_fused_joint['auc'], 2),
            'FPR': round(eval_fused_joint['fpr'], 2),
            'Latency_us': round(latency_us_per_flow, 2),
            'Throughput_fps': round(throughput_flows_sec, 0)
        }
    ])

    summary_df.to_csv(csv_file, index=False)
    try:
        summary_df.to_markdown(md_file, index=False)
    except Exception:
        pass

    print(f"\n[+] Results successfully exported to:")
    print(f"    - {csv_file}")
    print(f"    - {md_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid Ensemble NIDS Engine')
    parser.add_argument('--src', type=str, default='edge_iot', help='Source dataset')
    parser.add_argument('--tgt', type=str, default='ton_iot', help='Target dataset')
    parser.add_argument('--cpu', action='store_true', help='Force CPU instead of GPU')
    args = parser.parse_args()

    use_gpu = not args.cpu
    run_hybrid_ensemble_experiment(src_name=args.src, tgt_name=args.tgt, use_gpu=use_gpu)
