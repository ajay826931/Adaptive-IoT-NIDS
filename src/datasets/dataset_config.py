import os
from os.path import join

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
_BASE_DATA_PATH = os.path.join(_PROJECT_ROOT, "data")

dataset_config = {
    'iot_nidd': {  # class order
        'path': join(_BASE_DATA_PATH, 'uniform_label',
                     'iot-nidd_dwn10p.parquet'),
    },
    'edge_iot': {  # class order
        'path': join(_BASE_DATA_PATH, 'uniform_label',
                    'edge-iiot_dwn10p.parquet'),
    },
    'ton_iot': {  # class order
        'path': join(_BASE_DATA_PATH, 'uniform_label',
                     'ton-iot_dwn10p.parquet'),
    },
    'sample_iot': {
        'path': join(_BASE_DATA_PATH, 'uniform_label',
                     'sample_iot_dwn10p.parquet'),
    },
    'ddos_http': {  # New custom HTTP DDoS dataset from PCAP
        'path': join(_BASE_DATA_PATH, 'uniform_label',
                     'ddos_http_dwn10p.parquet'),
    }
}

min_max_config = {
    'PL': (-1,1500), 'IAT': (-1,60000), 'DIR': (0,1), 'WIN': (-1,65535),
}
