import os
import struct
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

# Feature normalization configuration (aligned with dataset_config.py)
MIN_MAX_CONFIG = {
    'PL': (-1.0, 1500.0),
    'IAT': (-1.0, 60000.0),
    'DIR': (0.0, 1.0),
    'WIN': (-1.0, 65535.0),
}

def scale_value(val, min_val, max_val):
    scaled = (float(val) - min_val) / (max_val - min_val)
    return float(np.clip(scaled, 0.0, 1.0))

def parse_pcap_to_biflows(pcap_path, num_pkts=10, max_flows=None):
    """
    Parses a raw PCAP file using pure Python struct and groups packets into 5-tuple bidirectional flows.
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    print(f"[+] Reading PCAP file: {pcap_path}")
    flows = defaultdict(list)
    flow_initiators = {} # Canonical key -> (client_ip, client_port)

    with open(pcap_path, 'rb') as f:
        global_hdr = f.read(24)
        if len(global_hdr) < 24:
            raise ValueError("Invalid PCAP file header.")
        magic, v_maj, v_min, thiszone, sigfigs, snaplen, network = struct.unpack('<IHHIIII', global_hdr)
        
        pkt_idx = 0
        while True:
            pkt_hdr = f.read(16)
            if len(pkt_hdr) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack('<IIII', pkt_hdr)
            data = f.read(incl_len)
            pkt_idx += 1
            timestamp = ts_sec * 1000.0 + ts_usec / 1000.0 # milliseconds

            # Ethernet parsing (linktype 1)
            if network == 1 and len(data) >= 34:
                eth_type = struct.unpack('!H', data[12:14])[0]
                if eth_type == 0x0800: # IPv4
                    ip_hdr = data[14:34]
                    proto = ip_hdr[9]
                    src_ip = '.'.join(str(b) for b in ip_hdr[12:16])
                    dst_ip = '.'.join(str(b) for b in ip_hdr[16:20])
                    
                    ihl = (ip_hdr[0] & 0x0f) * 4
                    trans_offset = 14 + ihl
                    
                    if proto == 6 and len(data) >= trans_offset + 20: # TCP
                        tcp_hdr = data[trans_offset:trans_offset+20]
                        src_port, dst_port = struct.unpack('!HH', tcp_hdr[0:4])
                        win_size = struct.unpack('!H', tcp_hdr[14:16])[0]
                        
                        # Canonical flow key
                        if (src_ip, src_port) < (dst_ip, dst_port):
                            flow_key = (src_ip, src_port, dst_ip, dst_port, proto)
                        else:
                            flow_key = (dst_ip, dst_port, src_ip, src_port, proto)
                        
                        if flow_key not in flow_initiators:
                            flow_initiators[flow_key] = (src_ip, src_port)
                            
                        client_ip, client_port = flow_initiators[flow_key]
                        direction = 0.0 if (src_ip == client_ip and src_port == client_port) else 1.0
                        
                        flows[flow_key].append({
                            'time': timestamp,
                            'length': orig_len,
                            'dir': direction,
                            'win': win_size
                        })

    print(f"[+] Processed {pkt_idx:,} packets. Found {len(flows):,} unique biflows.")
    return flows

def extract_flow_features(flows, num_pkts=10, label=1, train_ratio=0.8, seed=42):
    """
    Extracts 10-packet sequences of SCALED_PL, SCALED_IAT, SCALED_DIR, SCALED_WIN.
    """
    np.random.seed(seed)
    records = []
    
    for flow_key, packets in flows.items():
        if len(packets) < num_pkts:
            continue # Skip flows with fewer than required packets
            
        selected_pkts = packets[:num_pkts]
        
        pl_list = []
        iat_list = []
        dir_list = []
        win_list = []
        
        prev_time = selected_pkts[0]['time']
        for idx, pkt in enumerate(selected_pkts):
            # Packet Length
            pl_scaled = scale_value(pkt['length'], *MIN_MAX_CONFIG['PL'])
            pl_list.append(pl_scaled)
            
            # Inter-Arrival Time
            if idx == 0:
                iat_ms = 0.0
            else:
                iat_ms = max(0.0, pkt['time'] - prev_time)
            prev_time = pkt['time']
            iat_scaled = scale_value(iat_ms, *MIN_MAX_CONFIG['IAT'])
            iat_list.append(iat_scaled)
            
            # Direction
            dir_list.append(float(pkt['dir']))
            
            # Window size
            win_scaled = scale_value(pkt['win'], *MIN_MAX_CONFIG['WIN'])
            win_list.append(win_scaled)
            
        records.append({
            'SCALED_PL': pl_list,
            'SCALED_IAT': iat_list,
            'SCALED_DIR': dir_list,
            'SCALED_WIN': win_list,
            'ENC_LABEL': int(label),
        })
        
    print(f"[+] Qualified flows with >= {num_pkts} packets: {len(records):,}")
    
    df = pd.DataFrame(records)
    
    # Train / Test split
    num_samples = len(df)
    is_train = np.random.rand(num_samples) < train_ratio
    df['IS_TRAIN'] = is_train
    
    train_count = sum(is_train)
    test_count = num_samples - train_count
    print(f"[+] Dataset split: {train_count:,} Train samples ({train_count/num_samples*100:.1f}%), {test_count:,} Test samples ({test_count/num_samples*100:.1f}%)")
    
    return df

def convert_pcap(pcap_path, output_parquet, classes_txt=None, num_pkts=10, label=1, seed=1):
    output_dir = os.path.dirname(os.path.abspath(output_parquet))
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Parse attack flows from PCAP
    flows = parse_pcap_to_biflows(pcap_path, num_pkts=num_pkts)
    df_attack = extract_flow_features(flows, num_pkts=num_pkts, label=1, seed=seed)
    df_attack['LABEL'] = 'dos-http'
    
    # 2. Add benign baseline flows from edge-iiot or sample_iot for balanced evaluation
    benign_pool_path = os.path.join(output_dir, 'edge-iiot_dwn10p_prep1.parquet')
    if not os.path.exists(benign_pool_path):
        benign_pool_path = os.path.join(output_dir, 'sample_iot_dwn10p_prep1.parquet')
        
    if os.path.exists(benign_pool_path):
        print(f"[+] Sampling normal/benign baseline traffic from: {os.path.basename(benign_pool_path)}")
        df_pool = pd.read_parquet(benign_pool_path)
        # Class 0 is benign in standard label order
        df_benign = df_pool[df_pool['ENC_LABEL'] == 0].copy()
        # Subsample to match attack count
        n_samples = min(len(df_benign), len(df_attack))
        df_benign = df_benign.sample(n=n_samples, random_state=seed)
        df_benign['ENC_LABEL'] = 0
        df_benign['LABEL'] = 'benign'
        
        # Balance attack samples
        df_attack_sampled = df_attack.sample(n=n_samples, random_state=seed)
        
        df_combined = pd.concat([df_benign, df_attack_sampled], ignore_index=True)
        # Shuffle
        df_combined = df_combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        df_combined = df_attack

    # Ensure required columns
    cols = ['SCALED_PL', 'SCALED_IAT', 'SCALED_DIR', 'SCALED_WIN', 'ENC_LABEL', 'IS_TRAIN']
    df_final = df_combined[cols].copy()
    
    # Save standard parquet
    df_final.to_parquet(output_parquet, index=False)
    print(f"[+] Successfully saved dataset to: {output_parquet}")
    
    # Save prep0 and prep1 files for seed compatibility
    prep0_path = output_parquet.replace('.parquet', '_prep0.parquet')
    prep1_path = output_parquet.replace('.parquet', '_prep1.parquet')
    df_final.to_parquet(prep0_path, index=False)
    df_final.to_parquet(prep1_path, index=False)
    print(f"[+] Saved preprocessed cache files:\n    - {prep0_path}\n    - {prep1_path}")
    
    if classes_txt:
        with open(classes_txt, 'w') as f:
            f.write("benign\n")
            f.write("dos-http\n")
        print(f"[+] Created class mapping at: {classes_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw PCAP into preprocessed Parquet dataset for IoT-NIDS.")
    parser.add_argument("--pcap", type=str, 
                        default="../data/uniform_label/DDoS HTTP Flood Attacks.pcap",
                        help="Path to source PCAP file")
    parser.add_argument("--output", type=str,
                        default="../data/uniform_label/ddos_http_dwn10p.parquet",
                        help="Path to output Parquet file")
    parser.add_argument("--classes", type=str,
                        default="../data/uniform_label/classes_ddos_http_dwn10p.txt",
                        help="Path to output classes file")
    parser.add_argument("--num-pkts", type=int, default=10, help="Number of packets per flow window")
    parser.add_argument("--label", type=int, default=1, help="Encoded integer label (default 1 for dos-http)")
    
    args = parser.parse_args()
    convert_pcap(args.pcap, args.output, args.classes, args.num_pkts, args.label)
