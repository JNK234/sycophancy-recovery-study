#!/usr/bin/env python3
"""
ABOUTME: Generate figures for SimPO blog post v2 with statistical rigor
Generates: fig2_probe_transfer_v2.png, fig3_direction_similarity.png, fig4_ablation.png
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Load data
RESULTS_DIR = Path("/Users/jnk789/Developer/Sycophany-mini-research/results/probing/base-sft-dpo-simpo-ipo")

with open(RESULTS_DIR / "cross_model_transfer.json") as f:
    transfer_data = json.load(f)

with open(RESULTS_DIR / "direction_similarity.json") as f:
    similarity_data = json.load(f)

with open(RESULTS_DIR / "ablation.json") as f:
    ablation_data = json.load(f)

with open(RESULTS_DIR / "summary.json") as f:
    summary_data = json.load(f)

OUTPUT_DIR = Path("/Users/jnk789/Developer/Sycophany-mini-research/.claude/content/002-simpo-removing-the-anchor/figures")


def fig2_probe_transfer():
    """Probe transfer with confidence intervals and significance markers."""
    models = ['Base', 'DPO', 'SimPO']
    mean_aurocs = [
        transfer_data['sft_probe_on_base']['mean_auroc'],
        transfer_data['sft_probe_on_dpo']['mean_auroc'],
        transfer_data['sft_probe_on_simpo']['mean_auroc'],
    ]
    peak_aurocs = [
        transfer_data['sft_probe_on_base']['peak_auroc'],
        transfer_data['sft_probe_on_dpo']['peak_auroc'],
        transfer_data['sft_probe_on_simpo']['peak_auroc'],
    ]
    corrected_p = [
        transfer_data['sft_probe_on_base']['peak_corrected_p'],
        transfer_data['sft_probe_on_dpo']['peak_corrected_p'],
        transfer_data['sft_probe_on_simpo']['peak_corrected_p'],
    ]

    # CI from null distribution (approximate)
    null_95th = [
        transfer_data['sft_probe_on_base']['null_peak_95th'],
        transfer_data['sft_probe_on_dpo']['null_peak_95th'],
        transfer_data['sft_probe_on_simpo']['null_peak_95th'],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, mean_aurocs, width, label='Mean AUROC (36 layers)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, peak_aurocs, width, label='Peak AUROC', color='#e74c3c', alpha=0.8)

    # Add null 95th percentile line for each
    for i, (null, peak) in enumerate(zip(null_95th, peak_aurocs)):
        ax.hlines(null, x[i] + width/2 - 0.15, x[i] + width/2 + 0.15,
                  colors='gray', linestyles='dashed', linewidth=2)

    # Significance markers
    for i, (p, peak) in enumerate(zip(corrected_p, peak_aurocs)):
        if p < 0.05:
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*'
            ax.annotate(sig, xy=(x[i] + width/2, peak + 0.02), ha='center', fontsize=14, fontweight='bold')
        else:
            ax.annotate('n.s.', xy=(x[i] + width/2, peak + 0.02), ha='center', fontsize=10, color='gray')

    # Chance line
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Chance (0.5)')

    ax.set_ylabel('AUROC')
    ax.set_xlabel('Target Model')
    ax.set_title('SFT Probe Transfer: DPO Preserves, SimPO Reorganizes')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc='upper right')

    # Add value labels
    for bar, val in zip(bars1, mean_aurocs):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar, val in zip(bars2, peak_aurocs):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_probe_transfer_v2.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_probe_transfer_v2.pdf', bbox_inches='tight')
    print(f"Saved fig2_probe_transfer_v2.png")
    plt.close()


def fig3_direction_similarity():
    """Heatmap of cosine similarity between probe directions."""
    models = ['Base', 'SFT', 'DPO', 'SimPO']
    model_keys = ['base', 'sft', 'dpo', 'simpo']

    # Build similarity matrix (mean cosine)
    matrix = np.zeros((4, 4))
    for i, m1 in enumerate(model_keys):
        for j, m2 in enumerate(model_keys):
            key = f"{m1}_vs_{m2}" if i < j else f"{m2}_vs_{m1}"
            if i == j:
                matrix[i, j] = 1.0
            elif key in similarity_data:
                matrix[i, j] = similarity_data[key]['mean_cosine']
                matrix[j, i] = matrix[i, j]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Custom colormap centered at 0
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(matrix, annot=True, fmt='.3f', cmap=cmap, center=0,
                xticklabels=models, yticklabels=models, ax=ax,
                vmin=-0.4, vmax=1.0, annot_kws={'fontsize': 11})

    ax.set_title('Probe Direction Similarity (Cosine)\nHow similarly do models encode sycophancy?')

    # Highlight the key comparison
    ax.add_patch(plt.Rectangle((3, 1), 1, 1, fill=False, edgecolor='red', linewidth=3))
    ax.annotate('SFT vs SimPO:\nnearly orthogonal',
                xy=(3.5, 1.5), xytext=(4.2, 0.5),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_direction_similarity.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_direction_similarity.pdf', bbox_inches='tight')
    print(f"Saved fig3_direction_similarity.png")
    plt.close()


def fig4_ablation():
    """Bar chart showing ablation results: original -> ablated -> retrained."""
    models = ['Base', 'SFT', 'DPO', 'SimPO']
    model_keys = ['base', 'sft', 'dpo', 'simpo']

    original = [ablation_data[m]['auroc_original'] for m in model_keys]
    ablated = [ablation_data[m]['auroc_ablated'] for m in model_keys]
    retrained = [ablation_data[m]['auroc_retrained'] for m in model_keys]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, original, width, label='Original', color='#2ecc71', alpha=0.9)
    bars2 = ax.bar(x, ablated, width, label='After Ablation', color='#e74c3c', alpha=0.9)
    bars3 = ax.bar(x + width, retrained, width, label='Fresh Probe on Ablated', color='#3498db', alpha=0.9)

    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylabel('AUROC')
    ax.set_xlabel('Model')
    ax.set_title('Multi-Directional Sycophancy: Signal Persists After Removing Primary Direction')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')

    # Add value labels
    for bars, vals in [(bars1, original), (bars2, ablated), (bars3, retrained)]:
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 2), textcoords='offset points', ha='center', fontsize=8)

    # Add annotation about multi-directionality
    ax.annotate('All models recover signal\n→ Sycophancy is multi-directional',
                xy=(3.3, 0.73), fontsize=10, color='#3498db',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_ablation.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_ablation.pdf', bbox_inches='tight')
    print(f"Saved fig4_ablation.png")
    plt.close()


if __name__ == '__main__':
    print("Generating figures...")
    fig2_probe_transfer()
    fig3_direction_similarity()
    fig4_ablation()
    print("Done!")
