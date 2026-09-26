import os
import sys

# Ensure torch lib is on DLL path for Windows before importing numpy/pandas
venv_torch_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(venv_torch_lib) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(venv_torch_lib)
    except Exception:
        pass

import torch
import torch.nn.functional as F
import glob
import time
import argparse
import datetime
import numpy as np
import pandas as pd

from networks.stan_net import STANNet
from networks.network import LLL_Net
from datasets.dataset_config import dataset_config, min_max_config
from datasets.pcap_to_dataset import parse_pcap_to_biflows, extract_flow_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

# Attack Categorization Hierarchy (aligned with research paper Section 5.1)
ATTACK_CATEGORY_MAP = {
    'benign': ('NORMAL', 'Benign (Safe Traffic)'),
    'dos-http': ('MALICIOUS', 'DoS / DDoS Attack (HTTP Flood)'),
    'dos-icmp': ('MALICIOUS', 'DoS / DDoS Attack (ICMP Ping Flood)'),
    'dos-tcp-syn': ('MALICIOUS', 'DoS / DDoS Attack (TCP SYN Flood)'),
    'dos-tcp': ('MALICIOUS', 'DoS / DDoS Attack (TCP Flood)'),
    'dos-tcp-ack': ('MALICIOUS', 'DoS / DDoS Attack (TCP ACK Flood)'),
    'dos-udp': ('MALICIOUS', 'DoS / DDoS Attack (UDP Flood)'),
    'infog-scan-port': ('MALICIOUS', 'Reconnaissance / Port Scanning'),
    'infog-scan-os': ('MALICIOUS', 'Reconnaissance / OS Fingerprinting'),
    'infog-scan-host': ('MALICIOUS', 'Reconnaissance / Host Sweep'),
    'infog-scan-vuln': ('MALICIOUS', 'Reconnaissance / Vulnerability Scan'),
    'inject-http': ('MALICIOUS', 'Web Injection (HTTP Parameter/Header)'),
    'inject-sql': ('MALICIOUS', 'Database Injection (SQLi)'),
    'inject-xss': ('MALICIOUS', 'Script Injection (Cross-Site Scripting)'),
    'malwr-backd-tcp': ('MALICIOUS', 'Malware (TCP Backdoor / C2)'),
    'malwr-brutef-http': ('MALICIOUS', 'Credential Abuse (HTTP Brute-Force)'),
    'malwr-brutef-telnet': ('MALICIOUS', 'Credential Abuse (Telnet Brute-Force)'),
    'malwr-brutef-ftp': ('MALICIOUS', 'Credential Abuse (FTP Brute-Force)'),
    'malwr-ransomware': ('MALICIOUS', 'Ransomware / Encryptor Activity'),
    'mitm-arpspoof': ('MALICIOUS', 'Man-In-The-Middle (ARP Cache Poisoning)')
}

def find_best_checkpoint():
    """Finds the latest saved STANNet checkpoint in results/"""
    candidates = glob.glob(os.path.join(RESULTS_DIR, "*stannet*/models/*.ckpt"))
    if not candidates:
        candidates = glob.glob(os.path.join(RESULTS_DIR, "**/models/*.ckpt"), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def load_model(checkpoint_path=None, num_classes=14, device='cpu'):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        checkpoint_path = find_best_checkpoint()
        
    if not checkpoint_path:
        raise FileNotFoundError("No trained model checkpoint found in results/ directory.")
        
    print(f"[+] Loading model checkpoint: {os.path.relpath(checkpoint_path, BASE_DIR)}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Infer number of classes from head weights if available
    head_weights = [v for k, v in state_dict.items() if 'heads' in k and 'weight' in k]
    if head_weights:
        num_classes = head_weights[0].shape[0]
        
    net = STANNet(num_pkts=10, num_fields=4, num_classes=num_classes)
    lll_net = LLL_Net(net)
    lll_net.add_head(num_classes)
    
    # Load state dict
    lll_net.load_state_dict(state_dict, strict=False)
    lll_net.to(device)
    lll_net.eval()
    return lll_net, checkpoint_path, num_classes

def get_class_names(checkpoint_path, num_classes):
    """Detects class names from experiment folder or default edge-iiot classes"""
    exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    class_files = glob.glob(os.path.join(BASE_DIR, "data", "uniform_label", "classes_*.txt"))
    
    # Prefer edge-iiot (14 classes)
    for cf in class_files:
        if 'edge' in cf:
            with open(cf, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
                if len(lines) == num_classes:
                    return lines
                    
    # Fallback to any matching class length
    for cf in class_files:
        with open(cf, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            if len(lines) == num_classes:
                return lines
                
    return [f"Class_{i}" for i in range(num_classes)]

def prepare_input_data(target_input):
    """
    Handles input: can be dataset key ('edge_iot'), a .parquet path, or a raw .pcap path.
    """
    # 1. Dataset key check
    if target_input in dataset_config:
        parquet_path = dataset_config[target_input]['path']
        if not os.path.exists(parquet_path):
            parquet_path = os.path.join(BASE_DIR, parquet_path.replace('../', ''))
        prep_path = parquet_path.replace('.parquet', '_prep1.parquet')
        if os.path.exists(prep_path):
            parquet_path = prep_path
        print(f"[+] Loading dataset: {target_input} ({os.path.basename(parquet_path)})")
        return pd.read_parquet(parquet_path)
        
    # 2. File path check
    if not os.path.exists(target_input):
        candidate = os.path.join(BASE_DIR, target_input)
        if os.path.exists(candidate):
            target_input = candidate
        else:
            raise FileNotFoundError(f"Input file or dataset not found: {target_input}")
            
    # 3. PCAP conversion on-the-fly
    if target_input.lower().endswith('.pcap'):
        print(f"[+] Auto-parsing raw PCAP capture: {os.path.basename(target_input)}...")
        flows = parse_pcap_to_biflows(target_input, num_pkts=10)
        df = extract_flow_features(flows, num_pkts=10, label=0)
        return df
        
    # 4. Parquet file
    if target_input.lower().endswith('.parquet'):
        print(f"[+] Loading Parquet flows from: {os.path.basename(target_input)}")
        return pd.read_parquet(target_input)
        
    raise ValueError(f"Unsupported input format: {target_input}. Expected .parquet or .pcap")

def run_prediction(input_target, checkpoint=None, max_samples=1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load model and class names
    model, ckpt_path, num_classes = load_model(checkpoint, device=device)
    class_names = get_class_names(ckpt_path, num_classes)
    
    # 2. Prepare data
    df = prepare_input_data(input_target)
    if max_samples and len(df) > max_samples:
        print(f"[*] Subsampling {max_samples} flows for analysis (out of {len(df):,} total flows)...")
        df_sample = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        df_sample = df.reset_index(drop=True)
        
    # 3. Format tensor
    pl = np.vstack(df_sample['SCALED_PL'].values)
    iat = np.vstack(df_sample['SCALED_IAT'].values)
    dir_f = np.vstack(df_sample['SCALED_DIR'].values)
    win = np.vstack(df_sample['SCALED_WIN'].values)
    
    # Shape: (N, 1, 10, 4)
    data_tensor = np.stack([pl, iat, dir_f, win], axis=-1)
    data_tensor = np.expand_dims(data_tensor, axis=1)
    
    tensor_x = torch.tensor(data_tensor, dtype=torch.float32).to(device)
    
    # 4. Batch Inference
    print(f"[+] Running AI inference across {len(tensor_x):,} network flows...")
    predictions = []
    confidences = []
    
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(tensor_x), batch_size):
            batch_x = tensor_x[i:i+batch_size]
            outputs = model(batch_x)
            logits = torch.cat(outputs, dim=1)
            probs = F.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)
            predictions.extend(preds.cpu().numpy().tolist())
            confidences.extend(conf.cpu().numpy().tolist())
            
    # 5. Build Comprehensive Hierarchical Report
    results_list = []
    for idx in range(len(df_sample)):
        pred_idx = predictions[idx]
        attack_name = class_names[pred_idx] if pred_idx < len(class_names) else f"Attack_{pred_idx}"
        confidence = confidences[idx] * 100.0
        
        status, category = ATTACK_CATEGORY_MAP.get(attack_name, ('MALICIOUS', 'Unclassified Threat'))
        
        results_list.append({
            'Flow_ID': idx + 1,
            'Status': status,
            'Category': category,
            'Detected_Attack': attack_name,
            'Confidence_%': round(confidence, 2)
        })
        
    df_results = pd.DataFrame(results_list)
    
    # 6. Display Statistical Summary
    total_flows = len(df_results)
    malicious_count = (df_results['Status'] == 'MALICIOUS').sum()
    normal_count = (df_results['Status'] == 'NORMAL').sum()
    
    print("\n" + "=" * 70)
    print("         AUTOMATED IoT-NIDS TRAFFIC INTRUSION ANALYSIS REPORT        ")
    print("=" * 70)
    print(f" Total Network Flows Analyzed : {total_flows:,}")
    print(f" Normal / Benign Traffic      : {normal_count:,} ({normal_count/total_flows*100:.1f}%)")
    print(f" Malicious / Attack Detected  : {malicious_count:,} ({malicious_count/total_flows*100:.1f}%)")
    print("-" * 70)
    print(" Threat Category Breakdown:")
    cat_counts = df_results['Category'].value_counts()
    for cat, count in cat_counts.items():
        print(f"   * {cat:<40} : {count:>5} flows ({count/total_flows*100:>5.1f}%)")
    print("-" * 70)
    print(" Top Detected Attack Subtypes:")
    attack_counts = df_results['Detected_Attack'].value_counts()
    for att, count in attack_counts.items():
        print(f"   * {att:<25} : {count:>5} flows ({count/total_flows*100:>5.1f}%)")
    print("=" * 70)
    
    # 7. Print Sample Predictions Table
    print("\nSample Detections (Top 10 Flows):")
    print(df_results.head(10).to_string(index=False))
    
    # 8. Save CSV Report
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(PREDICTIONS_DIR, f"detection_report_{ts}.csv")
    df_results.to_csv(out_file, index=False)
    print(f"\n[+] Full detailed detection report saved to:\n    {out_file}\n")
    return out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Automated Attack Detection & Classification for IoT Traffic")
    parser.add_argument("input", nargs="?", default="edge_iot",
                        help="Dataset name ('edge_iot', 'ton_iot', 'ddos_http') OR path to a .parquet / .pcap file")
    parser.add_argument("--model", type=str, default=None, help="Path to custom model .ckpt file")
    parser.add_argument("--samples", type=int, default=1000, help="Maximum flows to analyze (default: 1000)")
    
    args = parser.parse_args()
    run_prediction(args.input, args.model, args.samples)
