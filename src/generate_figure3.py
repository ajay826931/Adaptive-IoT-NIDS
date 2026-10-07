import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

DATASETS = [
    {
        'id': 'iot_nidd',
        'title': '(a) IoT-NID',
        'parquet': 'iot-nidd_dwn10p_prep1.parquet',
        'classes': 'classes_iot-nidd_dwn10p.txt',
        'color': '#E67E22',
        'edgecolor': '#B95C00'
    },
    {
        'id': 'ton_iot',
        'title': '(b) TON_IoT',
        'parquet': 'ton-iot_dwn10p_prep1.parquet',
        'classes': 'classes_ton-iot_dwn10p.txt',
        'color': '#27AE60',
        'edgecolor': '#1E8449'
    },
    {
        'id': 'edge_iiot',
        'title': '(c) Edge-IIoT',
        'parquet': 'edge-iiot_dwn10p_prep1.parquet',
        'classes': 'classes_edge-iiot_dwn10p.txt',
        'color': '#2980B9',
        'edgecolor': '#1F618D'
    }
]

def generate_figure3():
    print("=" * 90)
    print("       GENERATING FIGURE 3: DATASET BIFLOW DISTRIBUTIONS (LOG SCALE)       ")
    print("=" * 90)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), sharey=True)
    summary_records = []

    for idx, dinfo in enumerate(DATASETS):
        ax = axes[idx]
        pq_path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', dinfo['parquet'])
        cls_path = os.path.join(PROJECT_ROOT, 'data', 'uniform_label', dinfo['classes'])

        class_names = [line.strip() for line in open(cls_path).readlines() if line.strip()]
        df = pd.read_parquet(pq_path)

        counts = df['ENC_LABEL'].value_counts()
        class_counts = []
        for c_idx, c_name in enumerate(class_names):
            cnt = counts.get(c_idx, 0)
            class_counts.append((c_name, cnt))
            summary_records.append({
                'Dataset': dinfo['title'].split(') ')[-1],
                'Class': c_name,
                'Biflow_Count': cnt,
                'Over_100_Samples': cnt > 100
            })

        # Sort descending by biflow count
        class_counts.sort(key=lambda x: x[1], reverse=True)
        names = [item[0] for item in class_counts]
        vals = [item[1] for item in class_counts]

        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, vals, color=dinfo['color'], edgecolor=dinfo['edgecolor'], width=0.68, linewidth=1.2, zorder=3)

        # Log scale
        ax.set_yscale('log')
        ax.set_ylim(1, 5e5)

        # Threshold line at 100 biflows
        ax.axhline(100, color='#C0392B', linestyle='--', linewidth=1.2, alpha=0.8, zorder=2, label='Threshold (#BF = 100)')

        # Annotations on top of bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            if height > 0:
                is_bold = height > 100
                ax.annotate(f'{int(height):,}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7.5,
                            fontweight='bold' if is_bold else 'normal',
                            color='#2C3E50',
                            rotation=90)

        # X-axis ticks with bold for >100 samples
        ax.set_xticks(x_pos)
        x_labels = [f"$\\mathbf{{{n}}}$" if v > 100 else n for n, v in zip(names, vals)]
        ax.set_xticklabels(x_labels, rotation=48, ha='right', fontsize=9)

        ax.set_title(dinfo['title'], fontsize=14, fontweight='bold', pad=12, color='#1A252F')
        ax.set_xlabel('Attack Class Label', fontsize=11, fontweight='semibold', labelpad=8)
        if idx == 0:
            ax.set_ylabel('# of Biflows (log scale)', fontsize=12, fontweight='bold', labelpad=8)

        ax.grid(axis='y', which='both', linestyle=':', alpha=0.6, zorder=1)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=9)

    plt.suptitle('Figure 3: Class Balance and Biflow Distribution Across IoT Benchmark Datasets',
                 fontsize=16, fontweight='bold', y=0.98, color='#0F172A')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_png = os.path.join(results_dir, 'figure3_biflow_distributions.png')
    out_csv = os.path.join(results_dir, 'figure3_distribution_summary.csv')

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(out_csv, index=False)

    print(f"\n[+] Figure 3 successfully generated at 300 DPI:")
    print(f"    - Image: {out_png}")
    print(f"    - CSV:   {out_csv}")

if __name__ == '__main__':
    generate_figure3()
