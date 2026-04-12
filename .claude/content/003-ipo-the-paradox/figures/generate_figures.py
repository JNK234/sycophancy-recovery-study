# ABOUTME: Generates publication-quality figures for Post 3 (IPO) blog post
# ABOUTME: 4 figures: behavioral comparison, probe transfer, layer AUROC curves, ablation comparison

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import numpy as np
import json
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(OUTDIR, '..', '..', '..', '..'))

# ── Publication-quality style ──
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#fafafa',
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
    'savefig.pad_inches': 0.3,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})

# ── Cohesive palette ──
COLORS = {
    'baseline': '#8E8E93',   # system gray
    'base':     '#8E8E93',   # alias for probing data keys
    'sft':      '#FF3B30',   # red
    'dpo':      '#007AFF',   # blue
    'simpo':    '#34C759',   # green
    'ipo':      '#FF9500',   # orange
}

LABELS = {
    'baseline': 'Baseline',
    'base': 'Base',
    'sft': 'SFT',
    'dpo': 'DPO',
    'simpo': 'SimPO',
    'ipo': 'IPO',
}


def add_value_label(ax, x, y, text, fontsize=8.5, color='#333', bold=True, offset=0.012):
    """Add a clean value label above a bar."""
    ax.text(x, y + offset, text, ha='center', va='bottom', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color=color,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='#fafafa')])


# ============================================================
# FIGURE 1: Growing behavioral comparison (5 models × 4 metrics)
# ============================================================

models = ['Baseline', 'SFT', 'DPO', 'SimPO', 'IPO']
colors = [COLORS['baseline'], COLORS['sft'], COLORS['dpo'], COLORS['simpo'], COLORS['ipo']]

metrics = {
    'Aggregate\nSycophancy': [0.256, 0.467, 0.268, 0.176, 0.281],
    'Answer\nSycophancy':    [0.393, 0.604, 0.447, 0.275, 0.417],
    'Flip Rate':             [0.259, 0.600, 0.264, 0.104, 0.257],
    'Feedback\nSycophancy':  [0.115, 0.196, 0.095, 0.058, 0.170],
}

fig, axes = plt.subplots(1, 4, figsize=(17, 5.5))
fig.suptitle('IPO recovers sycophancy to near-DPO levels — but with steeper capability damage',
             fontsize=14, fontweight='bold', y=1.03, color='#222')

for ax, (metric_name, values) in zip(axes, metrics.items()):
    bars = ax.bar(models, values, color=colors, edgecolor='white', linewidth=0.8, width=0.65,
                  zorder=3)
    ax.set_title(metric_name, fontsize=10.5, fontweight='bold', color='#444', pad=10)
    ax.set_ylim(0, max(values) * 1.28)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    for bar, val in zip(bars, values):
        add_value_label(ax, bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.3f}', fontsize=8)

    ax.tick_params(axis='x', rotation=35, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Light horizontal gridlines
    ax.yaxis.grid(True, linestyle=':', alpha=0.3, zorder=0)

axes[0].set_ylabel('Rate', fontsize=11, color='#444')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig1_behavioral_comparison.png'))
plt.close()
print("✓ Figure 1: behavioral comparison")


# ============================================================
# FIGURE 2: Probe transfer bar chart with significance markers
# ============================================================

with open(os.path.join(proj_root, 'results/probing/base-sft-dpo-simpo-ipo/summary.json')) as f:
    probing = json.load(f)

transfer_models = ['Base', 'DPO', 'SimPO', 'IPO']
transfer_keys = ['sft_probe_on_base', 'sft_probe_on_dpo', 'sft_probe_on_simpo', 'sft_probe_on_ipo']
transfer_colors = [COLORS['baseline'], COLORS['dpo'], COLORS['simpo'], COLORS['ipo']]

mean_aurocs = [probing['cross_model_transfer'][k]['mean_auroc'] for k in transfer_keys]
corrected_ps = [probing['cross_model_transfer'][k]['peak_corrected_p'] for k in transfer_keys]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(transfer_models, mean_aurocs, color=transfer_colors, edgecolor='white',
              linewidth=0.8, width=0.55, zorder=3)

# Reference lines
ax.axhline(y=0.5, color='#999', linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)
ax.text(3.45, 0.505, 'chance', fontsize=8, color='#999', ha='right', style='italic')

control_mean = probing['control']['overall_mean']
ax.axhline(y=control_mean, color='#bbb', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
ax.text(3.45, control_mean + 0.005, f'noise floor ({control_mean:.3f})', fontsize=8, color='#bbb',
        ha='right', style='italic')

# Labels with significance
for bar, val, p in zip(bars, mean_aurocs, corrected_ps):
    if p < 0.01:
        sig_text = f'p = {p:.3f} ***'
        sig_color = '#222'
    elif p < 0.05:
        sig_text = f'p = {p:.3f} *'
        sig_color = '#555'
    else:
        sig_text = f'p = {p:.3f}  n.s.'
        sig_color = '#888'

    add_value_label(ax, bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', fontsize=10, offset=0.018)
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.045,
            sig_text, ha='center', va='bottom', fontsize=7.5, color=sig_color, style='italic',
            path_effects=[pe.withStroke(linewidth=2, foreground='#fafafa')])

ax.set_ylabel('Mean AUROC (SFT probe -> target model)', fontsize=11, color='#444')
ax.set_title('SFT sycophancy probe fails on IPO — transfer 0.365, the lowest of all methods',
             fontsize=13, fontweight='bold', color='#222', pad=12)
ax.set_ylim(0.25, 0.85)
ax.yaxis.grid(True, linestyle=':', alpha=0.3, zorder=0)
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig2_probe_transfer.png'))
plt.close()
print("✓ Figure 2: probe transfer")


# ============================================================
# FIGURE 3: Layer-by-layer AUROC curves (IPO layer-3 anomaly)
# ============================================================

with open(os.path.join(proj_root, 'results/probing/base-sft-dpo-simpo-ipo/per_model.json')) as f:
    per_model = json.load(f)

fig, ax = plt.subplots(figsize=(11, 5.5))

model_keys = ['base', 'sft', 'dpo', 'simpo', 'ipo']
linewidths = [1.3, 1.3, 1.5, 1.5, 2.8]
alphas =     [0.55, 0.55, 0.7, 0.7, 1.0]
zorders =    [1, 1, 2, 2, 4]

for key, lw, alpha, zo in zip(model_keys, linewidths, alphas, zorders):
    layers = sorted([int(l) for l in per_model[key]['per_layer'].keys()])
    aurocs = [per_model[key]['per_layer'][str(l)]['auroc'] for l in layers]
    ax.plot(layers, aurocs, '-', label=LABELS[key], color=COLORS[key],
            linewidth=lw, zorder=zo, alpha=alpha)

    # Peak marker
    peak_layer = per_model[key]['peak_layer']
    peak_auroc = per_model[key]['peak_auroc']
    ms = 10 if key == 'ipo' else 5
    ax.plot(peak_layer, peak_auroc, 'o', color=COLORS[key], markersize=ms,
            zorder=zo + 1, markeredgecolor='white', markeredgewidth=1.2)

# Annotate IPO layer-3 peak with a clean callout
ipo_peak = per_model['ipo']['peak_auroc']
ax.annotate(
    'IPO peaks at layer 3\n(all others: layers 17-22)',
    xy=(3, ipo_peak), xytext=(10, ipo_peak + 0.045),
    fontsize=10.5, fontweight='bold', color=COLORS['ipo'],
    arrowprops=dict(arrowstyle='->', color=COLORS['ipo'], lw=2,
                    connectionstyle='arc3,rad=0.2'),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff8f0', edgecolor=COLORS['ipo'],
              alpha=0.9, linewidth=1.2)
)

ax.axhline(y=0.5, color='#bbb', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Layer', fontsize=11, color='#444')
ax.set_ylabel('AUROC (own-model probe)', fontsize=11, color='#444')
ax.set_title('IPO moves sycophancy-relevant processing from late layers to layer 3',
             fontsize=13, fontweight='bold', color='#222', pad=12)
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9, edgecolor='#ddd', ncol=2)
ax.set_xlim(-0.5, 35.5)
ax.yaxis.grid(True, linestyle=':', alpha=0.25, zorder=0)
ax.tick_params(labelsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig3_layer_auroc_curves.png'))
plt.close()
print("✓ Figure 3: layer AUROC curves")


# ============================================================
# FIGURE 4: Ablation comparison (original vs retrained)
# ============================================================

ablation = probing['ablation']
abl_keys = ['base', 'sft', 'dpo', 'simpo', 'ipo']

original = [ablation[k]['auroc_original'] for k in abl_keys]
retrained = [ablation[k]['auroc_retrained'] for k in abl_keys]

x = np.arange(len(abl_keys))
width = 0.32

fig, ax = plt.subplots(figsize=(10, 5.5))

# Original bars (solid)
bars1 = ax.bar(x - width/2, original, width, label='Original AUROC (peak layer)',
               color=[COLORS[k] for k in abl_keys], edgecolor='white', linewidth=0.8, zorder=3)

# Retrained bars (hatched, lighter)
bars2 = ax.bar(x + width/2, retrained, width, label='Retrained after ablating top direction',
               color=[COLORS[k] for k in abl_keys], edgecolor='white', linewidth=0.8,
               alpha=0.45, hatch='///', zorder=3)

# Value labels
for bar, val in zip(bars1, original):
    add_value_label(ax, bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', fontsize=8.5)
for bar, val in zip(bars2, retrained):
    add_value_label(ax, bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', fontsize=8.5)

# Drop annotations (delta between original and retrained)
for i, (o, r) in enumerate(zip(original, retrained)):
    delta = o - r
    color = COLORS[abl_keys[i]]
    if abs(delta) < 0.005:
        label = 'Δ ≈ 0'
        fw = 'bold'
    else:
        label = f'Δ = {delta:+.3f}'
        fw = 'normal'
    ax.text(x[i], max(o, r) + 0.045, label, ha='center', va='bottom',
            fontsize=8.5, fontweight=fw, color=color,
            path_effects=[pe.withStroke(linewidth=2, foreground='#fafafa')])

# IPO callout
ax.annotate(
    'IPO: removing the top direction\nhas zero effect (0.811 -> 0.814)',
    xy=(4 + width/2, retrained[4] + 0.01),
    xytext=(3.0, 0.88),
    fontsize=10, fontweight='bold', color=COLORS['ipo'],
    arrowprops=dict(arrowstyle='->', color=COLORS['ipo'], lw=1.8,
                    connectionstyle='arc3,rad=-0.2'),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff8f0', edgecolor=COLORS['ipo'],
              alpha=0.9, linewidth=1.2)
)

ax.set_ylabel('AUROC', fontsize=11, color='#444')
ax.set_title("IPO's sycophancy signal is fully distributed — no single direction matters",
             fontsize=13, fontweight='bold', color='#222', pad=12)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[k] for k in abl_keys], fontsize=10)
ax.legend(loc='lower left', fontsize=9, framealpha=0.9, edgecolor='#ddd')
ax.set_ylim(0.6, 0.95)
ax.yaxis.grid(True, linestyle=':', alpha=0.25, zorder=0)
ax.tick_params(labelsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig4_ablation_comparison.png'))
plt.close()
print("✓ Figure 4: ablation comparison")

print("\n✅ All 4 figures generated successfully in:", OUTDIR)
