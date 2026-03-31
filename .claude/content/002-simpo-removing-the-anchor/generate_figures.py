# ABOUTME: Generates figures for Blog Post 002 (SimPO — Removing the Anchor)
# ABOUTME: Two figures: behavioral comparison (4 models) and probe transfer AUROC

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

# Output directory
fig_dir = os.path.dirname(os.path.abspath(__file__)) + "/figures"
os.makedirs(fig_dir, exist_ok=True)

# ---------- DATA (from results/eval/*/summary.json and results/probing/*/summary.json) ----------

models = ["Base", "SFT", "DPO", "SimPO"]
model_colors = {
    "Base": "#7f8c8d",    # gray
    "SFT": "#e74c3c",     # red
    "DPO": "#3498db",     # blue
    "SimPO": "#2ecc71",   # green
}

# Behavioral metrics
aggregate_syc = [0.256, 0.467, 0.268, 0.176]
flip_rate =     [0.259, 0.600, 0.264, 0.104]
feedback_syc =  [0.115, 0.196, 0.095, 0.058]
answer_syc =    [0.393, 0.604, 0.447, 0.365]
poems_syc =     [0.297, 0.443, 0.238, 0.007]

# Probing (2931-prompt full run)
probe_transfer_models = ["Base", "DPO", "SimPO"]
probe_transfer_values = [0.611, 0.677, 0.503]
probe_transfer_colors = ["#7f8c8d", "#3498db", "#2ecc71"]

# Direction similarity (2931-prompt full run)
cosine_sft_dpo = 0.210
cosine_sft_simpo = 0.082

# ---------- STYLE SETUP ----------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})


# ============================================================================
# FIGURE 1: Behavioral Comparison (4 models, grouped bars)
# ============================================================================
def fig1_behavioral():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ("Aggregate Sycophancy", aggregate_syc, "SimPO goes below baseline"),
        ("Flip Rate (Are You Sure?)", flip_rate, "SimPO: 10% vs baseline 26%"),
        ("Poem Sycophancy", poems_syc, "30% baseline → 0.7% SimPO"),
    ]

    x = np.arange(len(models))
    width = 0.6

    for ax, (title, values, subtitle) in zip(axes, metrics):
        bars = ax.bar(x, values, width, color=[model_colors[m] for m in models],
                      edgecolor='white', linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            y_pos = bar.get_height() + 0.01
            label = f"{val:.1%}" if val >= 0.01 else f"{val:.1%}"
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        # Add baseline reference line
        ax.axhline(y=values[0], color='#7f8c8d', linestyle='--', alpha=0.4, linewidth=1)

        # Subtitle annotation
        ax.text(0.5, -0.15, subtitle, transform=ax.transAxes,
                ha='center', fontsize=9, color='#555555', style='italic')

        # Set y-axis limit with headroom for labels
        ax.set_ylim(0, max(values) * 1.25)

    fig.suptitle("SimPO recovers sycophancy below baseline — DPO only reaches baseline",
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(f"{fig_dir}/fig1_behavioral_comparison.png", dpi=200, bbox_inches='tight')
    print(f"Saved fig1_behavioral_comparison.png")
    plt.close()


# ============================================================================
# FIGURE 2: Probe Transfer AUROC (SFT probe → each model)
# ============================================================================
def fig2_probe_transfer():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 2]})

    # --- Left panel: Transfer AUROC ---
    x = np.arange(len(probe_transfer_models))
    width = 0.5

    bars = ax1.bar(x, probe_transfer_values, width,
                   color=probe_transfer_colors,
                   edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, probe_transfer_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Chance line at 0.5
    ax1.axhline(y=0.5, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=1.5, label='Chance (0.5)')
    ax1.text(len(probe_transfer_models) - 0.5, 0.505, 'chance',
             color='#e74c3c', fontsize=9, alpha=0.8, ha='right')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"SFT → {m}" for m in probe_transfer_models], fontsize=10)
    ax1.set_ylabel("Transfer AUROC", fontsize=11)
    ax1.set_title("SFT sycophancy probe transfer\n(2,931 prompts)", fontsize=12, fontweight='bold')
    ax1.set_ylim(0.35, 0.75)

    # Annotations
    ax1.annotate("Pattern persists", xy=(1, 0.677), xytext=(1.3, 0.72),
                fontsize=9, color='#3498db', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.2))
    ax1.annotate("At chance —\npattern gone", xy=(2, 0.503), xytext=(2.3, 0.44),
                fontsize=9, color='#2ecc71', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.2))

    # --- Right panel: Cosine Similarity ---
    cosine_models = ["SFT vs DPO", "SFT vs SimPO"]
    cosine_values = [cosine_sft_dpo, cosine_sft_simpo]
    cosine_colors = ["#3498db", "#2ecc71"]

    x2 = np.arange(len(cosine_models))
    bars2 = ax2.bar(x2, cosine_values, 0.45, color=cosine_colors,
                    edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars2, cosine_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Reference lines
    ax2.axhline(y=0.0, color='#555555', linestyle='-', alpha=0.3, linewidth=0.5)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cosine_models, fontsize=10)
    ax2.set_ylabel("Cosine Similarity", fontsize=11)
    ax2.set_title("Probe direction similarity\n(sycophancy encoding angle)", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 0.30)

    # Annotations
    ax2.text(0, cosine_sft_dpo / 2, "Partially\nshared", ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')
    ax2.text(1, cosine_sft_simpo + 0.03, "Nearly\northogonal", ha='center', va='bottom',
             fontsize=9, color='#2ecc71', fontweight='bold')

    fig.suptitle("DPO suppresses sycophancy at the output — SimPO reorganizes it internally",
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(f"{fig_dir}/fig2_probe_transfer.png", dpi=200, bbox_inches='tight')
    print(f"Saved fig2_probe_transfer.png")
    plt.close()


# ============================================================================
# RUN ALL
# ============================================================================
if __name__ == "__main__":
    fig1_behavioral()
    fig2_probe_transfer()
    print("All figures generated.")
