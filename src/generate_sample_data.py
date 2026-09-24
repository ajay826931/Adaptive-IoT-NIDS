import pandas as pd
import numpy as np
import os

def create_sample_dataset():
    num_samples = 1500
    num_pkts = 10
    
    # Generate random arrays for each feature per sample
    # PL: Packet Length (0 to 1500)
    pl_data = [np.random.randint(0, 1500, size=num_pkts) for _ in range(num_samples)]
    
    # IAT: Inter-Arrival Time (0 to 60000 ms)
    iat_data = [np.random.randint(0, 60000, size=num_pkts) for _ in range(num_samples)]
    
    # DIR: Direction (0 or 1)
    dir_data = [np.random.randint(0, 2, size=num_pkts) for _ in range(num_samples)]
    
    # WIN: Window size (0 to 65535)
    win_data = [np.random.randint(0, 65535, size=num_pkts) for _ in range(num_samples)]
    
    # Generate labels
    labels = np.random.choice(["Normal", "DDoS", "Malware"], size=num_samples)
    
    # Optional padding indicator for the network dataset
    feat_pad = [0 for _ in range(num_samples)]
    
    # Create the dataframe
    df = pd.DataFrame({
        'PL': pl_data,
        'IAT': iat_data,
        'DIR': dir_data,
        'WIN': win_data,
        'FEAT_PAD': feat_pad,
        'LABEL': labels
    })
    
    # Ensure directory exists
    out_dir = '../data/uniform_label'
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as parquet
    out_path = os.path.join(out_dir, 'sample_iot_dwn10p.parquet')
    df.to_parquet(out_path)
    
    print(f"Successfully generated {num_samples} samples of IoT dataset at {out_path}!")

if __name__ == '__main__':
    create_sample_dataset()
