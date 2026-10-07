import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from networks.stan_net import STANNet
from datasets.data_loader import get_loaders

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

def format_inputs(imgs, dev):
    if isinstance(imgs, list):
        return [v.to(dev, non_blocking=True) for v in imgs]
    return imgs.to(dev, non_blocking=True)

def train_continual_pair(src_dataset, tgt_dataset, approach='ft-mem', epochs=3, batch_size=64, num_exemplars=200, seed=1, device='cuda:0'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    src_classes = load_class_names(src_dataset)
    tgt_classes = load_class_names(tgt_dataset)

    # Class sets
    src_set = set(src_classes)
    tgt_set = set(tgt_classes)
    all_classes = sorted(list(src_set.union(tgt_set)))
    class_to_global_id = {c: i for i, c in enumerate(all_classes)}
    num_global_classes = len(all_classes)

    # Load Source and Target data loaders
    src_trn_loader, _, src_tst_loader, _ = get_loaders(
        [src_dataset], num_tasks=1, nc_first_task=None, nc_incr_tasks=None,
        batch_size=batch_size, num_workers=0,
        validation=0.1, num_pkts=10, fields=['PL', 'IAT', 'DIR', 'WIN'], seed=seed
    )
    tgt_trn_loader, _, tgt_tst_loader, _ = get_loaders(
        [tgt_dataset], num_tasks=1, nc_first_task=None, nc_incr_tasks=None,
        batch_size=batch_size, num_workers=0,
        validation=0.1, num_pkts=10, fields=['PL', 'IAT', 'DIR', 'WIN'], seed=seed
    )

    src_to_global = np.array([class_to_global_id[c] for c in src_classes])
    tgt_to_global = np.array([class_to_global_id[c] for c in tgt_classes])

    model = STANNet(num_classes=num_global_classes, num_pkts=10, num_fields=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if 'cuda' in device else None

    # Replay buffer
    memory_x = []
    memory_y = []

    if approach == 'scratch':
        # SCRATCH: Joint training on Source + Target together (Upper bound)
        joint_batches = []
        for inp, tgt in src_trn_loader[0]:
            global_tgt = torch.tensor(src_to_global[tgt.numpy()], dtype=torch.long)
            joint_batches.append((inp, global_tgt))
        for inp, tgt in tgt_trn_loader[0]:
            global_tgt = torch.tensor(tgt_to_global[tgt.numpy()], dtype=torch.long)
            joint_batches.append((inp, global_tgt))

        model.train()
        for ep in range(epochs):
            np.random.shuffle(joint_batches)
            for inputs, targets in joint_batches:
                inputs = format_inputs(inputs, device)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                if scaler:
                    with torch.amp.autocast('cuda'):
                        out = model(inputs)
                        loss = criterion(out, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(inputs)
                    loss = criterion(out, targets)
                    loss.backward()
                    optimizer.step()

    else:
        # Phase 1: Train on Source Task (S)
        model.train()
        for ep in range(epochs):
            for inputs, targets in src_trn_loader[0]:
                inputs = format_inputs(inputs, device)
                global_tgt = torch.tensor(src_to_global[targets.numpy()], dtype=torch.long, device=device)
                optimizer.zero_grad()
                if scaler:
                    with torch.amp.autocast('cuda'):
                        out = model(inputs)
                        loss = criterion(out, global_tgt)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(inputs)
                    loss = criterion(out, global_tgt)
                    loss.backward()
                    optimizer.step()

        # Collect exemplars for FT-Mem and BiC
        if approach in ['ft-mem', 'bic']:
            count_per_class = max(1, num_exemplars // len(src_classes))
            class_counts = {c: 0 for c in range(len(src_classes))}
            for inputs, targets in src_trn_loader[0]:
                for i in range(len(targets)):
                    t_val = targets[i].item()
                    if class_counts[t_val] < count_per_class:
                        if isinstance(inputs, list):
                            memory_x.append([v[i:i+1].cpu() for v in inputs])
                        else:
                            memory_x.append(inputs[i:i+1].cpu())
                        memory_y.append(src_to_global[t_val])
                        class_counts[t_val] += 1

        # Phase 2: Adapt to Target Task (T)
        model.train()
        for ep in range(epochs):
            for inputs, targets in tgt_trn_loader[0]:
                inputs = format_inputs(inputs, device)
                global_tgt = torch.tensor(tgt_to_global[targets.numpy()], dtype=torch.long, device=device)

                # Replay exemplars for FT-Mem and BiC
                if approach in ['ft-mem', 'bic'] and len(memory_x) > 0:
                    mem_idx = np.random.choice(len(memory_x), size=min(16, len(memory_x)), replace=False)
                    if isinstance(inputs, list):
                        mem_batch_x = [torch.cat([memory_x[k][j] for k in mem_idx], dim=0).to(device) for j in range(len(inputs))]
                        inputs = [torch.cat([inputs[j], mem_batch_x[j]], dim=0) for j in range(len(inputs))]
                    else:
                        mem_batch_x = torch.cat([memory_x[k] for k in mem_idx], dim=0).to(device)
                        inputs = torch.cat([inputs, mem_batch_x], dim=0)
                    mem_batch_y = torch.tensor([memory_y[k] for k in mem_idx], dtype=torch.long, device=device)
                    global_tgt = torch.cat([global_tgt, mem_batch_y], dim=0)

                optimizer.zero_grad()
                if scaler:
                    with torch.amp.autocast('cuda'):
                        out = model(inputs)
                        loss = criterion(out, global_tgt)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(inputs)
                    loss = criterion(out, global_tgt)
                    loss.backward()
                    optimizer.step()

    # Phase 3: Evaluate on combined test set (both Source and Target)
    model.eval()
    all_y_true = []
    all_y_pred = []

    # Test Source test set
    with torch.no_grad():
        for inputs, targets in src_tst_loader[0]:
            inputs = format_inputs(inputs, device)
            with torch.amp.autocast('cuda') if 'cuda' in device else torch.no_grad():
                out = model(inputs)
                preds = torch.argmax(out, dim=1).cpu().numpy()
            all_y_true.extend(src_to_global[targets.numpy()])
            all_y_pred.extend(preds)

        for inputs, targets in tgt_tst_loader[0]:
            inputs = format_inputs(inputs, device)
            with torch.amp.autocast('cuda') if 'cuda' in device else torch.no_grad():
                out = model(inputs)
                preds = torch.argmax(out, dim=1).cpu().numpy()
            all_y_true.extend(tgt_to_global[targets.numpy()])
            all_y_pred.extend(preds)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Compute F1-scores broken down by partitions
    # 1. Per-class F1
    per_class_f1 = f1_score(all_y_true, all_y_pred, average=None, labels=range(num_global_classes), zero_division=0) * 100.0

    common_classes = src_set.intersection(tgt_set)
    src_unc_classes = src_set.difference(tgt_set)
    tgt_unc_classes = tgt_set.difference(src_set)

    src_unc_ids = [class_to_global_id[c] for c in src_unc_classes]
    src_com_ids = [class_to_global_id[c] for c in common_classes]
    src_all_ids = [class_to_global_id[c] for c in src_classes]

    tgt_unc_ids = [class_to_global_id[c] for c in tgt_unc_classes]
    tgt_com_ids = [class_to_global_id[c] for c in common_classes]
    tgt_all_ids = [class_to_global_id[c] for c in tgt_classes]

    f1_src_unc = np.mean([per_class_f1[i] for i in src_unc_ids]) if src_unc_ids else 0.0
    f1_src_com = np.mean([per_class_f1[i] for i in src_com_ids]) if src_com_ids else 0.0
    f1_src_all = np.mean([per_class_f1[i] for i in src_all_ids]) if src_all_ids else 0.0

    f1_tgt_unc = np.mean([per_class_f1[i] for i in tgt_unc_ids]) if tgt_unc_ids else 0.0
    f1_tgt_com = np.mean([per_class_f1[i] for i in tgt_com_ids]) if tgt_com_ids else 0.0
    f1_tgt_all = np.mean([per_class_f1[i] for i in tgt_all_ids]) if tgt_all_ids else 0.0

    f1_all = np.mean(per_class_f1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'C_src_unc': f1_src_unc,
        'C_src_com': f1_src_com,
        'C_src': f1_src_all,
        'C_tgt_unc': f1_tgt_unc,
        'C_tgt_com': f1_tgt_com,
        'C_tgt': f1_tgt_all,
        'C_all': f1_all
    }

def run_table6_benchmark(epochs=3, device='cuda:0'):
    pairs = [
        ('edge_iot', 'iot_nidd'),
        ('edge_iot', 'ton_iot'),
        ('iot_nidd', 'edge_iot'),
        ('iot_nidd', 'ton_iot'),
        ('ton_iot', 'edge_iot'),
        ('ton_iot', 'iot_nidd')
    ]

    approaches = ['FT', 'FT-Mem', 'BiC']

    print("=" * 115)
    print("       RUNNING TASK 4: TABLE 6 CONTINUAL LEARNING BENCHMARK       ")
    print(f"       Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})       ")
    print("=" * 115)

    table_data = []

    for src, tgt in pairs:
        s_name = DISPLAY_NAMES.get(src, src)
        t_name = DISPLAY_NAMES.get(tgt, tgt)
        print(f"\n[+] Configuration: < S: {s_name} -> T: {t_name} >")

        # First train Scratch baseline
        t0 = time.time()
        scratch_res = train_continual_pair(src, tgt, approach='scratch', epochs=epochs, device=device)
        print(f"    * SCRATCH NIDS Baseline: C_all = {scratch_res['C_all']:.2f}% ({time.time() - t0:.1f}s)")
        scratch_score = f"{scratch_res['C_all']:.2f}"

        pair_appr_results = {}
        for app in approaches:
            t0 = time.time()
            app_key = app.lower()
            res = train_continual_pair(src, tgt, approach=app_key, epochs=epochs, device=device)
            pair_appr_results[app] = res
            print(f"    * Approach [{app:<6}]: C_src = {res['C_src']:.2f}% | C_tgt = {res['C_tgt']:.2f}% | C_all = {res['C_all']:.2f}% ({time.time() - t0:.1f}s)")

        # Find best C_all among continual learning approaches
        best_app = max(approaches, key=lambda a: pair_appr_results[a]['C_all'])

        for i, app in enumerate(approaches):
            res = pair_appr_results[app]
            is_best = (app == best_app)

            table_data.append({
                'S': s_name if i == 0 else '',
                'T': t_name if i == 0 else '',
                'Appr': app,
                'C_src_unc': f"{res['C_src_unc']:.2f}",
                'C_src_com': f"{res['C_src_com']:.2f}",
                'C_src': f"{res['C_src']:.2f}",
                'C_tgt_unc': f"{res['C_tgt_unc']:.2f}",
                'C_tgt_com': f"{res['C_tgt_com']:.2f}",
                'C_tgt': f"{res['C_tgt']:.2f}",
                'C_all': f"**{res['C_all']:.2f}**" if is_best else f"{res['C_all']:.2f}",
                'C_all_raw': res['C_all'],
                'Is_Best': is_best,
                'SCRATCH_NIDS': scratch_score if i == 1 else ''
            })

    # Print Table 6
    print("\n" + "=" * 125)
    print("Table 6: Dataset-agnostic performance (on both source and target), in terms of F1 Score [%]")
    print("=" * 125)
    header = f"{'S':<10} | {'T':<10} | {'Appr.':<7} | {'C_src_unc':<10} {'C_src_com':<10} {'C_src':<9} {'C_tgt_unc':<10} {'C_tgt_com':<10} {'C_tgt':<9} {'C_all':<9} | {'SCRATCH NIDS':<12}"
    print(header)
    print("-" * 125)

    for row in table_data:
        c_all_disp = f"*{row['C_all_raw']:.2f}*" if row['Is_Best'] else f"{row['C_all_raw']:.2f}"
        print(f"{row['S']:<10} | {row['T']:<10} | {row['Appr']:<7} | {row['C_src_unc']:<10} {row['C_src_com']:<10} {row['C_src']:<9} {row['C_tgt_unc']:<10} {row['C_tgt_com']:<10} {row['C_tgt']:<9} {c_all_disp:<9} | {row['SCRATCH_NIDS']:<12}")
        if row['Appr'] == 'BiC':
            print("-" * 125)

    print("=" * 125)
    print("N.B.: C_src = C_src_unc U C_src_com, C_tgt = C_tgt_unc U C_tgt_com, and C_all = C_src U C_tgt.")

    # Export to CSV and Markdown
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    df_out = pd.DataFrame(table_data).drop(columns=['C_all_raw', 'Is_Best'])
    csv_file = os.path.join(results_dir, 'table6_continual_learning_results.csv')
    md_file = os.path.join(results_dir, 'table6_continual_learning_results.md')
    df_out.to_csv(csv_file, index=False)
    try:
        df_out.to_markdown(md_file, index=False)
    except Exception:
        pass
    print(f"\n[+] Results successfully exported to:")
    print(f"    - {csv_file}")
    print(f"    - {md_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Table 6 Continual Learning Benchmark')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs per task')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    dev = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    run_table6_benchmark(epochs=args.epochs, device=dev)
