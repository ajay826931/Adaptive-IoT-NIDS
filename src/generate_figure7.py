import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from hybrid_ensemble import load_parquet_dataset, extract_multidimensional_features, DISPLAY_NAMES

def generate_figure7():
    print("=" * 95)
    print("   GENERATING FIGURE 7: SEQUENTIAL FEATURE HEATMAPS & 2D LATENT MANIFOLD   ")
    print("=" * 95)

    # 1. Load samples from all three datasets
    dsets = ['edge_iot', 'iot_nidd', 'ton_iot']
    data_dict = {}
    print("\n[Step 1] Loading datasets and computing packet statistics...")
    for d in dsets:
        df = load_parquet_dataset(d)
        data_dict[d] = df
        print(f"    * {DISPLAY_NAMES.get(d, d)}: {len(df)} samples loaded")

    # 2. Extract Average 10-Packet Feature Matrix per Dataset
    # Fields: PL, IAT, DIR, WIN
    field_names = ['PL (Packet Length)', 'IAT (Inter-Arrival Time)', 'DIR (Packet Direction)', 'WIN (Window Size)']
    field_keys = ['SCALED_PL', 'SCALED_IAT', 'SCALED_DIR', 'SCALED_WIN']
    colormaps = ['Blues', 'Oranges', 'Purples', 'Greens']

    dset_labels = [DISPLAY_NAMES[d] for d in dsets]

    # Create master figure: Top rows = Feature Heatmaps, Bottom = 2D Latent Manifold
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.3], hspace=0.32, wspace=0.28)

    print("\n[Step 2] Plotting Section 1.4.1 Sequential Feature Heatmaps (10-packet biflow)...")
    for f_idx, (fname, fkey, cmap_name) in enumerate(zip(field_names, field_keys, colormaps)):
        ax = fig.add_subplot(gs[0, f_idx])

        # Compute average sequence across first 10 packets for each dataset
        heatmap_mat = np.zeros((3, 10))
        for r_idx, d in enumerate(dsets):
            seqs = np.vstack(data_dict[d][fkey].values)[:, :10]
            heatmap_mat[r_idx, :] = np.mean(seqs, axis=0)

        im = ax.imshow(heatmap_mat, cmap=cmap_name, aspect='auto', interpolation='nearest')

        # Cell annotations
        for r in range(3):
            for c in range(10):
                val = heatmap_mat[r, c]
                color = 'white' if val > (np.max(heatmap_mat) * 0.6) else 'black'
                ax.text(c, r, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color, fontweight='medium')

        ax.set_xticks(range(10))
        ax.set_xticklabels([f"P{i+1}" for i in range(10)], fontsize=8.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels(dset_labels, fontsize=10, fontweight='semibold')
        ax.set_title(fname, fontsize=11, fontweight='bold', pad=9, color='#1E293B')
        ax.set_xlabel('Packet Index in Biflow', fontsize=9.5)

        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.18, shrink=0.85)
        cbar.ax.tick_params(labelsize=8)

    # 3. Compute 2D Latent Space Manifold (t-SNE on multi-dimensional features)
    print("\n[Step 3] Computing Section 1.4.2 2D Latent Manifold (t-SNE / PCA) on GPU/CPU...")
    sample_size = 1200
    features_list = []
    labels_dataset = []
    palette = {'Edge-IIoT': '#2980B9', 'IoT-NID': '#E67E22', 'TON_IoT': '#27AE60'}

    for d in dsets:
        df = data_dict[d]
        if len(df) > sample_size:
            sampled_df = df.sample(n=sample_size, random_state=42)
        else:
            sampled_df = df
        X_feats, _, _ = extract_multidimensional_features(sampled_df, num_pkts=10)
        features_list.append(X_feats)
        labels_dataset.extend([DISPLAY_NAMES[d]] * len(X_feats))

    X_combined = np.vstack(features_list)
    labels_dataset = np.array(labels_dataset)

    # Dimensionality reduction: PCA first to 30 dims for stability, then t-SNE to 2 dims
    pca = PCA(n_components=min(30, X_combined.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_combined)

    tsne = TSNE(n_components=2, perplexity=35, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_pca)

    # Subplot for t-SNE Latent Space Manifold (Spanning cols 0 to 2)
    ax_manifold = fig.add_subplot(gs[1, :2])
    for d_name in ['Edge-IIoT', 'IoT-NID', 'TON_IoT']:
        mask = (labels_dataset == d_name)
        ax_manifold.scatter(
            X_embedded[mask, 0], X_embedded[mask, 1],
            c=palette[d_name], label=d_name, alpha=0.65, s=18, edgecolors='none'
        )

    ax_manifold.set_title('Figure 7(b): 2D Latent Manifold Projection (Domain Overlap & Separation)',
                          fontsize=12, fontweight='bold', pad=10, color='#1E293B')
    ax_manifold.set_xlabel('t-SNE Manifold Dimension 1', fontsize=10, fontweight='semibold')
    ax_manifold.set_ylabel('t-SNE Manifold Dimension 2', fontsize=10, fontweight='semibold')
    ax_manifold.legend(loc='upper right', frameon=True, fontsize=10, title="IoT Domain", title_fontsize='10')
    ax_manifold.grid(True, linestyle=':', alpha=0.5)

    # Subplot for Explained Variance / Feature Importance (Cols 2 to 4)
    ax_var = fig.add_subplot(gs[1, 2:])
    pca_full = PCA(n_components=15, random_state=42)
    pca_full.fit(X_combined)
    exp_var = pca_full.explained_variance_ratio_ * 100.0
    cum_var = np.cumsum(exp_var)

    x_pcs = np.arange(1, 16)
    ax_var.bar(x_pcs, exp_var, color='#3B82F6', alpha=0.8, edgecolor='#1D4ED8', width=0.6, label='Individual Variance [%]')
    ax_var.plot(x_pcs, cum_var, color='#EF4444', marker='o', linewidth=2, label='Cumulative Variance [%]')

    for x_i, c_v in zip(x_pcs, cum_var):
        if x_i in [1, 3, 5, 10, 15]:
            ax_var.annotate(f"{c_v:.1f}%", xy=(x_i, c_v), xytext=(0, 6), textcoords="offset points",
                            ha='center', fontsize=8.5, fontweight='bold', color='#B91C1C')

    ax_var.set_xticks(x_pcs)
    ax_var.set_xlabel('Principal Feature Components', fontsize=10, fontweight='semibold')
    ax_var.set_ylabel('Explained Variance Ratio [%]', fontsize=10, fontweight='semibold')
    ax_var.set_title('Figure 7(c): Multi-Dimensional Feature Spectrum & Variance Retention',
                     fontsize=12, fontweight='bold', pad=10, color='#1E293B')
    ax_var.legend(loc='center right', frameon=True, fontsize=9.5)
    ax_var.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle('Figure 7: Sequential Feature Heatmaps and Latent Space Manifold Representation',
                 fontsize=16, fontweight='bold', y=0.98, color='#0F172A')

    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_png = os.path.join(results_dir, 'figure7_features_and_latent_manifold.png')

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[+] Figure 7 successfully generated at 300 DPI:")
    print(f"    - Image: {out_png}")

if __name__ == '__main__':
    generate_figure7()
