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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from networks.stan_net import STANNet
from datasets.data_loader import get_loaders

# Canonical 20 Classes (1 to 20) exactly aligned with reference paper Fig. 4(g)
CANONICAL_CLASSES = [
    'benign',               # 1
    'dos-http',             # 2
    'dos-icmp',             # 3
    'dos-tcp',              # 4
    'dos-tcp-ack',          # 5
    'dos-tcp-syn',          # 6
    'dos-udp',              # 7
    'infog-scan-host',      # 8
    'infog-scan-os',        # 9
    'infog-scan-port',      # 10
    'infog-scan-vuln',      # 11
    'inject-http',          # 12
    'inject-sql',           # 13
    'inject-xss',           # 14
    'malwr-backd-tcp',      # 15
    'malwr-brutef-ftp',     # 16
    'malwr-brutef-http',    # 17
    'malwr-brutef-telnet',  # 18
    'malwr-ransomware',     # 19
    'mitm-arpspoof'         # 20
]

CANONICAL_MAP = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

CLASS_FILE_MAP = {
    'edge_iot': 'classes_edge-iiot_dwn10p.txt',
    'edge-iot': 'classes_edge-iiot_dwn10p.txt',
    'edge-iiot': 'classes_edge-iiot_dwn10p.txt',
    'iot_nidd': 'classes_iot-nidd_dwn10p.txt',
    'iot-nidd': 'classes_iot-nidd_dwn10p.txt',
    'ton_iot': 'classes_ton-iot_dwn10p.txt',
    'ton-iot': 'classes_ton-iot_dwn10p.txt',
}

def load_class_names(dataset_name):
    fname = CLASS_FILE_MAP.get(dataset_name, f"classes_{dataset_name}_dwn10p.txt")
    classes_path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', fname)
    if not os.path.exists(classes_path):
        classes_path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', f"classes_{dataset_name.replace('_', '-')}_dwn10p.txt")
    return [line.strip() for line in open(classes_path).readlines() if line.strip()]

def train_source_model(src_dataset, epochs=3, batch_size=64, seed=1, device='cuda:0'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trn_loader, _, _, taskcla = get_loaders(
        [src_dataset], num_tasks=1, nc_first_task=None, nc_incr_tasks=None,
        batch_size=batch_size, num_workers=0,
        validation=0.1, num_pkts=10, fields=['PL', 'IAT', 'DIR', 'WIN'], seed=seed
    )
    
    num_classes = taskcla[0][1]
    model = STANNet(num_classes=num_classes, num_pkts=10, num_fields=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler('cuda') if 'cuda' in device else None
    train_dl = trn_loader[0]
    
    model.train()
    for ep in range(epochs):
        for inputs, targets in train_dl:
            if isinstance(inputs, list):
                inputs = [v.to(device, non_blocking=True) for v in inputs]
            else:
                inputs = inputs.to(device, non_blocking=True)
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
                
    return model

def evaluate_transfer(model, src_dataset, tgt_dataset, seed=1, device='cuda:0'):
    src_classes = load_class_names(src_dataset)
    tgt_classes = load_class_names(tgt_dataset)
    
    # Load target test set
    _, _, tst_loader, _ = get_loaders(
        [tgt_dataset], num_tasks=1, nc_first_task=None, nc_incr_tasks=None,
        batch_size=64, num_workers=0,
        validation=0.1, num_pkts=10, fields=['PL', 'IAT', 'DIR', 'WIN'], seed=seed
    )
    
    test_dl = tst_loader[0]
    model.eval()
    
    all_tgt_true = []
    all_src_pred = []
    
    with torch.no_grad():
        for inputs, targets in test_dl:
            if isinstance(inputs, list):
                inputs = [v.to(device, non_blocking=True) for v in inputs]
            else:
                inputs = inputs.to(device, non_blocking=True)
            if hasattr(targets, 'cpu'):
                targets_np = targets.cpu().numpy()
            else:
                targets_np = np.array(targets)
                
            with torch.amp.autocast('cuda') if 'cuda' in device else torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
            all_tgt_true.extend(targets_np)
            all_src_pred.extend(preds)
            
    # Build 20x20 transfer matrix
    # Matrix shape: (20 target classes, 20 source classes)
    matrix = np.full((20, 20), np.nan)
    
    # Identify which canonical classes are active in source and target
    src_active_idx = [CANONICAL_MAP[c] for c in src_classes if c in CANONICAL_MAP]
    tgt_active_idx = [CANONICAL_MAP[c] for c in tgt_classes if c in CANONICAL_MAP]
    
    # Initialize active grid cells with 0.0
    for r in tgt_active_idx:
        for c in src_active_idx:
            matrix[r, c] = 0.0
            
    # Count predictions
    all_tgt_true = np.array(all_tgt_true)
    all_src_pred = np.array(all_src_pred)
    
    for local_tgt_cls_id in np.unique(all_tgt_true):
        tgt_cname = tgt_classes[local_tgt_cls_id]
        if tgt_cname not in CANONICAL_MAP:
            continue
        canon_tgt_id = CANONICAL_MAP[tgt_cname]
        
        # Filter samples for this target class
        mask = (all_tgt_true == local_tgt_cls_id)
        total_samples = np.sum(mask)
        if total_samples == 0:
            continue
            
        preds_for_cls = all_src_pred[mask]
        for local_src_pred_id in preds_for_cls:
            src_cname = src_classes[local_src_pred_id]
            if src_cname in CANONICAL_MAP:
                canon_src_id = CANONICAL_MAP[src_cname]
                matrix[canon_tgt_id, canon_src_id] += 1.0
                
        # Normalize to percentage (0 - 100%)
        for c in src_active_idx:
            matrix[canon_tgt_id, c] = (matrix[canon_tgt_id, c] / total_samples) * 100.0
            
    return matrix, src_classes, tgt_classes

def plot_fig4_transferability(matrices_dict, output_path='results/figure4_transferability.png'):
    fig = plt.figure(figsize=(20, 11), dpi=300)
    
    # Grid layout: 2 rows of 3 heatmap subplots
    pairs_info = [
        ('a', ('edge_iot', 'iot_nidd'), '⟨ Edge-IIoT, IoT-NID ⟩'),
        ('b', ('edge_iot', 'ton_iot'),  '⟨ Edge-IIoT, TON_IoT ⟩'),
        ('c', ('ton_iot', 'edge_iot'),  '⟨ TON_IoT, Edge-IIoT ⟩'),
        ('d', ('ton_iot', 'iot_nidd'),  '⟨ TON_IoT, IoT-NID ⟩'),
        ('e', ('iot_nidd', 'ton_iot'),  '⟨ IoT-NID, TON_IoT ⟩'),
        ('f', ('iot_nidd', 'edge_iot'), '⟨ IoT-NID, Edge-IIoT ⟩')
    ]
    
    cmap = plt.cm.YlOrBr.copy()
    cmap.set_bad(color='#e6e6e6')  # Gray for missing/padding classes
    norm = mcolors.Normalize(vmin=0, vmax=100)
    
    # Add shared colorbar at the top
    cbar_ax = fig.add_axes([0.35, 0.94, 0.30, 0.02])
    cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_ticks([0, 20, 40, 60, 80, 100])
    cb.ax.tick_params(labelsize=10)
    
    # 2 rows x 4 columns grid (with column 4 for table encoding on bottom row)
    gs = fig.add_gridspec(2, 4, top=0.90, bottom=0.08, left=0.04, right=0.98, wspace=0.30, hspace=0.35)
    
    plot_positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1)
    ]
    
    for idx, (tag, pair_key, title) in enumerate(pairs_info):
        r, c = plot_positions[idx]
        ax = fig.add_subplot(gs[r, c])
        
        mat = matrices_dict[pair_key]
        masked_mat = np.ma.masked_invalid(mat)
        
        im = ax.imshow(masked_mat, cmap=cmap, norm=norm, origin='upper')
        
        # Highlight diagonal correspondence with red boxes
        for i in range(20):
            # If both source and target have this class active
            if not np.isnan(mat[i, i]):
                rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=1.5, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
        ax.set_xticks(range(20))
        ax.set_xticklabels(range(1, 21), fontsize=7)
        ax.set_yticks(range(20))
        ax.set_yticklabels(range(1, 21), fontsize=7)
        
        ax.set_xlabel('Source Dataset', fontsize=8, labelpad=2)
        ax.set_ylabel('Target Dataset', fontsize=8, labelpad=2)
        ax.set_title(f'({tag}) {title}', fontsize=10, pad=6, fontweight='bold')
        
        # Subtle grid
        ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
        ax.grid(which='minor', color='#d0d0d0', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        
    # Bottom right: Classes Encoding Table (g)
    ax_table = fig.add_subplot(gs[1, 2:])
    ax_table.axis('off')
    
    table_data = [
        ["1", "benign", "11", "infog-scan-vuln"],
        ["2", "dos-http", "12", "inject-http"],
        ["3", "dos-icmp", "13", "inject-sql"],
        ["4", "dos-tcp", "14", "inject-xss"],
        ["5", "dos-tcp-ack", "15", "malwr-backd-tcp"],
        ["6", "dos-tcp-syn", "16", "malwr-brutef-ftp"],
        ["7", "dos-udp", "17", "malwr-brutef-http"],
        ["8", "infog-scan-host", "18", "malwr-brutef-telnet"],
        ["9", "infog-scan-os", "19", "malwr-ransomware"],
        ["10", "infog-scan-port", "20", "mitm-arpspoof"],
    ]
    
    col_labels = [r"N$^o$", "Classes", r"N$^o$", "Classes"]
    t = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='left')
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1.0, 1.4)
    
    # Style table header
    for (i, j), cell in t.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f0f0f0')
            cell.set_linewidth(1.2)
        else:
            cell.set_linewidth(0.4)
            if j in [0, 2]:
                cell.set_text_props(fontweight='bold')
                
    ax_table.set_title("(g) Classes Encoding", fontsize=11, fontweight='bold', pad=10)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[+] High-Resolution Figure 4 successfully saved to: {output_path}")

def run_cross_dataset_experiment(epochs=3, device='cuda:0'):
    pairs = [
        ('edge_iot', 'iot_nidd'),
        ('edge_iot', 'ton_iot'),
        ('ton_iot', 'edge_iot'),
        ('ton_iot', 'iot_nidd'),
        ('iot_nidd', 'ton_iot'),
        ('iot_nidd', 'edge_iot')
    ]
    
    print("=" * 100)
    print(f"       RUNNING TASK 3: CROSS-DATASET DOMAIN SHIFT BENCHMARK (6 PAIRS)       ")
    print(f"       Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})       ")
    print("=" * 100)
    
    trained_models = {}
    matrices = {}
    
    unique_sources = list(set([p[0] for p in pairs]))
    
    # 1. Train source models once
    for src in unique_sources:
        print(f"\n[+] Training Source Baseline Model on: {src.upper()} ({epochs} epochs on GPU)...")
        t0 = time.time()
        model = train_source_model(src, epochs=epochs, device=device)
        print(f"    -> Trained in {time.time() - t0:.1f}s")
        trained_models[src] = model
        
    # 2. Evaluate all 6 transfer pairs
    summary_rows = []
    
    for src, tgt in pairs:
        print(f"\n[+] Evaluating Cross-Dataset Domain Shift: < {src.upper()} -> {tgt.upper()} >...")
        t0 = time.time()
        mat, src_classes, tgt_classes = evaluate_transfer(trained_models[src], src, tgt, device=device)
        matrices[(src, tgt)] = mat
        
        # Calculate diagonal accuracy (shared class transferability)
        shared = []
        for i in range(20):
            if not np.isnan(mat[i, i]):
                shared.append(mat[i, i])
        avg_diagonal = np.mean(shared) if shared else 0.0
        print(f"    -> Transfer Evaluation Complete in {time.time() - t0:.1f}s | Shared Class Mean Acc: {avg_diagonal:.2f}%")
        
        summary_rows.append({
            'Source': src,
            'Target': tgt,
            'Shared_Classes_Count': len(shared),
            'Mean_Diagonal_Accuracy': f"{avg_diagonal:.2f}%"
        })
        
    # 3. Export Figure 4
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    fig_path = os.path.join(results_dir, 'figure4_cross_dataset_transferability.png')
    plot_fig4_transferability(matrices, fig_path)
    
    # 4. Export CSV Summary
    df_sum = pd.DataFrame(summary_rows)
    csv_path = os.path.join(results_dir, 'figure4_transfer_summary.csv')
    df_sum.to_csv(csv_path, index=False)
    print(f"[+] Transfer Summary exported to: {csv_path}")
    
    print("\n" + "=" * 65)
    print("       CROSS-DATASET TRANSFER SUMMARY (DIAGONAL ACC)")
    print("=" * 65)
    print(df_sum.to_string(index=False))
    print("=" * 65)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-Dataset Transfer Benchmark')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs for source training')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    dev = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    run_cross_dataset_experiment(epochs=args.epochs, device=dev)
