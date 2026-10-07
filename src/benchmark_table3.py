import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from networks.stan_net import STANNet
from networks.lopez17cnn import Lopez17CNN
from datasets.data_loader import get_loaders

def get_macro_category(cls_name):
    cls_name = cls_name.lower().strip()
    if cls_name == 'benign':
        return 'benign'
    elif cls_name.startswith('dos') or 'flood' in cls_name:
        return 'dos'
    elif cls_name.startswith('infog') or 'scan' in cls_name:
        return 'scanning'
    elif cls_name.startswith('inject') or 'xss' in cls_name or 'sql' in cls_name:
        return 'injection'
    elif cls_name.startswith('malwr') or 'backdoor' in cls_name or 'ransomware' in cls_name or 'brutef' in cls_name:
        return 'malware'
    elif cls_name.startswith('mitm') or 'spoof' in cls_name:
        return 'mitm'
    return 'other'

def compute_pauc(y_true, y_scores, max_fpr=0.1):
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        stop = np.searchsorted(fpr, max_fpr, 'right')
        x_interp = [fpr[stop - 1], max_fpr]
        y_interp = [tpr[stop - 1], np.interp(max_fpr, [fpr[stop - 1], fpr[stop]], [tpr[stop - 1], tpr[stop]])]
        x = np.append(fpr[:stop], max_fpr)
        y = np.append(tpr[:stop], y_interp[1])
        partial_auc = auc(x, y) / max_fpr
        return partial_auc * 100.0
    except Exception:
        return roc_auc_score(y_true, y_scores) * 100.0

def train_and_evaluate(dataset_name, network_name='STANNet', epochs=5, batch_size=64, seed=1, device='cuda:0'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Load data
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(
        [dataset_name], num_tasks=1, nc_first_task=None, nc_incr_tasks=None,
        batch_size=batch_size, num_workers=0,
        validation=0.1, num_pkts=10, fields=['PL', 'IAT', 'DIR', 'WIN'], seed=seed
    )
    
    num_classes = taskcla[0][1]
    
    # 2. Build model
    if network_name.lower() == 'stannet':
        model = STANNet(num_classes=num_classes, num_pkts=10, num_fields=4)
    else:
        model = Lopez17CNN(num_classes=num_classes, num_pkts=10, num_fields=4)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    train_dl = trn_loader[0]
    test_dl = tst_loader[0]
    
    # 3. Train on GPU
    model.train()
    scaler = torch.amp.GradScaler('cuda') if 'cuda' in device else None
    
    def format_inputs(imgs, dev):
        if isinstance(imgs, list):
            return [v.to(dev, non_blocking=True) for v in imgs]
        return imgs.to(dev, non_blocking=True)

    for epoch in range(epochs):
        for batch in train_dl:
            inputs, targets = batch
            inputs = format_inputs(inputs, device)
            targets = targets.to(device, dtype=torch.long, non_blocking=True)
            
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
    # 4. Evaluate on test set
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_dl:
            inputs, targets = batch
            inputs = format_inputs(inputs, device)
            if hasattr(targets, 'cpu'):
                targets_np = targets.cpu().numpy()
            else:
                targets_np = np.array(targets)
            
            with torch.amp.autocast('cuda') if 'cuda' in device else torch.no_grad():
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
            all_targets.extend(targets_np)
            all_probs.extend(probs)
            
    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    y_pred = np.argmax(y_probs, axis=1)
    
    CLASS_FILE_MAP = {
        'edge_iot': 'classes_edge-iiot_dwn10p.txt',
        'edge-iot': 'classes_edge-iiot_dwn10p.txt',
        'edge-iiot': 'classes_edge-iiot_dwn10p.txt',
        'iot_nidd': 'classes_iot-nidd_dwn10p.txt',
        'iot-nidd': 'classes_iot-nidd_dwn10p.txt',
        'ton_iot': 'classes_ton-iot_dwn10p.txt',
        'ton-iot': 'classes_ton-iot_dwn10p.txt',
        'sample_iot': 'classes_sample_iot_dwn10p.txt',
        'ddos_http': 'classes_ddos_http_dwn10p.txt',
    }
    fname = CLASS_FILE_MAP.get(dataset_name, f"classes_{dataset_name}_dwn10p.txt")
    classes_path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', fname)
    if not os.path.exists(classes_path):
        classes_path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', f"classes_{dataset_name.replace('_', '-')}_dwn10p.txt")
    class_names = [line.strip() for line in open(classes_path).readlines() if line.strip()]
    
    benign_idx = 0
    for idx, cname in enumerate(class_names):
        if cname.lower() == 'benign':
            benign_idx = idx
            break
            
    # --- METRIC 1: Binary MD ---
    y_bin_true = (y_true != benign_idx).astype(int)
    prob_malicious = 1.0 - y_probs[:, benign_idx]
    y_bin_pred = (y_pred != benign_idx).astype(int)
    
    bin_f1 = f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0
    bin_recall = recall_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0
    bin_acc = accuracy_score(y_bin_true, y_bin_pred) * 100.0
    try:
        bin_auc = roc_auc_score(y_bin_true, prob_malicious) * 100.0
    except Exception:
        bin_auc = 100.0
    bin_pauc = compute_pauc(y_bin_true, prob_malicious, max_fpr=0.1)
    
    # --- METRIC 2: Multiclass MD (Category) ---
    cat_names = sorted(list(set(get_macro_category(c) for c in class_names)))
    cat_to_id = {c: i for i, c in enumerate(cat_names)}
    class_to_cat_id = np.array([cat_to_id[get_macro_category(c)] for c in class_names])
    
    y_cat_true = class_to_cat_id[y_true]
    y_cat_pred = class_to_cat_id[y_pred]
    
    cat_f1 = f1_score(y_cat_true, y_cat_pred, average='macro', zero_division=0) * 100.0
    cat_recall = recall_score(y_cat_true, y_cat_pred, average='macro', zero_division=0) * 100.0
    cat_acc = accuracy_score(y_cat_true, y_cat_pred) * 100.0
    
    # --- METRIC 3: Multiclass MD (Type / Fine-Grained) ---
    type_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    type_recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    type_acc = accuracy_score(y_true, y_pred) * 100.0
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        'bin_f1': bin_f1, 'bin_rec': bin_recall, 'bin_acc': bin_acc, 'bin_auc': bin_auc, 'bin_pauc': bin_pauc,
        'cat_f1': cat_f1, 'cat_rec': cat_recall, 'cat_acc': cat_acc,
        'type_f1': type_f1, 'type_rec': type_recall, 'type_acc': type_acc
    }

def run_table3_benchmark(datasets=None, runs=3, epochs=5, network='STANNet', device='cuda:0'):
    if datasets is None:
        datasets = ['edge_iot', 'iot_nidd', 'ton_iot']

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print("=" * 115)
    print(f"       RUNNING TABLE 3 BENCHMARK (S = T, Across {runs} Runs on {network})       ")
    print(f"       Device: {device} ({gpu_name})       ")
    print("=" * 115)
    
    results = {}
    display_names = {
        'edge_iot': 'Edge-IIoT',
        'iot_nidd': 'IoT-NID',
        'ton_iot': 'TON_IoT'
    }
    
    for dset in datasets:
        dname = display_names.get(dset, dset)
        print(f"\n[+] Evaluating Dataset: {dname} ({runs} runs, {epochs} epochs each)...")
        metric_runs = []
        
        for r in range(1, runs + 1):
            t0 = time.time()
            metrics = train_and_evaluate(dset, network_name=network, epochs=epochs, seed=r, device=device)
            elapsed = time.time() - t0
            print(f"   * Run {r}/{runs} ({elapsed:.1f}s) -> Bin F1: {metrics['bin_f1']:.2f}% | Cat F1: {metrics['cat_f1']:.2f}% | Type F1: {metrics['type_f1']:.2f}%")
            metric_runs.append(metrics)
            
        summary = {}
        for k in metric_runs[0].keys():
            vals = [m[k] for m in metric_runs]
            summary[k] = (np.mean(vals), np.std(vals))
        results[dname] = summary
        
    # Print Table 3
    print("\n" + "=" * 125)
    print("Table 3: NIDS trained and tested within the same network environment, i.e., S = T. (avg ± std)")
    print("=" * 125)
    h1 = f"{'Dataset':<12} | {'Binary MD':<48} | {'Multiclass MD (category)':<30} | {'Multiclass MD (type)':<30}"
    h2 = f"{'':<12} | {'F1 [%]':<9} {'Recall':<9} {'Acc [%]':<9} {'AUC [%]':<9} {'PAUC [%]':<9} | {'F1 [%]':<9} {'Recall':<9} {'Acc [%]':<9} | {'F1 [%]':<9} {'Recall':<9} {'Acc [%]':<9}"
    print(h1)
    print(h2)
    print("-" * 125)
    
    rows_for_csv = []
    
    for dname, s in results.items():
        row_str = (
            f"{dname:<12} | "
            f"{s['bin_f1'][0]:5.2f}±{s['bin_f1'][1]:.2f} "
            f"{s['bin_rec'][0]:5.2f}±{s['bin_rec'][1]:.2f} "
            f"{s['bin_acc'][0]:5.2f}±{s['bin_acc'][1]:.2f} "
            f"{s['bin_auc'][0]:5.2f}±{s['bin_auc'][1]:.2f} "
            f"{s['bin_pauc'][0]:5.2f}±{s['bin_pauc'][1]:.2f} | "
            f"{s['cat_f1'][0]:5.2f}±{s['cat_f1'][1]:.2f} "
            f"{s['cat_rec'][0]:5.2f}±{s['cat_rec'][1]:.2f} "
            f"{s['cat_acc'][0]:5.2f}±{s['cat_acc'][1]:.2f} | "
            f"{s['type_f1'][0]:5.2f}±{s['type_f1'][1]:.2f} "
            f"{s['type_rec'][0]:5.2f}±{s['type_rec'][1]:.2f} "
            f"{s['type_acc'][0]:5.2f}±{s['type_acc'][1]:.2f}"
        )
        print(row_str)
        
        rows_for_csv.append({
            'Dataset': dname,
            'Binary_F1': f"{s['bin_f1'][0]:.2f}±{s['bin_f1'][1]:.2f}",
            'Binary_Recall': f"{s['bin_rec'][0]:.2f}±{s['bin_rec'][1]:.2f}",
            'Binary_Acc': f"{s['bin_acc'][0]:.2f}±{s['bin_acc'][1]:.2f}",
            'Binary_AUC': f"{s['bin_auc'][0]:.2f}±{s['bin_auc'][1]:.2f}",
            'Binary_PAUC': f"{s['bin_pauc'][0]:.2f}±{s['bin_pauc'][1]:.2f}",
            'Category_F1': f"{s['cat_f1'][0]:.2f}±{s['cat_f1'][1]:.2f}",
            'Category_Recall': f"{s['cat_rec'][0]:.2f}±{s['cat_rec'][1]:.2f}",
            'Category_Acc': f"{s['cat_acc'][0]:.2f}±{s['cat_acc'][1]:.2f}",
            'Type_F1': f"{s['type_f1'][0]:.2f}±{s['type_f1'][1]:.2f}",
            'Type_Recall': f"{s['type_rec'][0]:.2f}±{s['type_rec'][1]:.2f}",
            'Type_Acc': f"{s['type_acc'][0]:.2f}±{s['type_acc'][1]:.2f}",
        })
        
    print("=" * 125)
    
    # Save CSV and text
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    df_res = pd.DataFrame(rows_for_csv)
    df_res.to_csv(os.path.join(results_dir, 'table3_benchmark_results.csv'), index=False)
    print(f"\n[+] Results successfully exported to: {os.path.join(results_dir, 'table3_benchmark_results.csv')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Table 3 Benchmark Generator')
    parser.add_argument('--datasets', nargs='+', default=['edge_iot', 'iot_nidd', 'ton_iot'])
    parser.add_argument('--runs', type=int, default=3, help='Number of random seed runs')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per run')
    parser.add_argument('--network', type=str, default='STANNet', choices=['STANNet', 'Lopez17CNN'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    run_table3_benchmark(datasets=args.datasets, runs=args.runs, epochs=args.epochs, network=args.network, device=device)
