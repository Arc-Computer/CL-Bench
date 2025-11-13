#!/usr/bin/env python3
"""
Generate Diagram 1: Consistency Gap Bridge
Academic-quality visualization showing baseline capability vs Atlas's bridging of the consistency gap
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
})

# Data from baseline evaluation results (artifacts/baselines/)
# Based on 400-conversation standardized evaluation subset
models = {
    'Claude 4.5 Sonnet': {
        'capability': 76.0,      # Conversations w/ ≥1 tool success
        'turn_level': 45.8,      # Turn-level tool success
        'strict': 7.0,           # Strict success (all turns)
        'color': '#666666',      # Dark gray
        'linestyle': '--',
        'marker': 'o',
        'zorder': 2
    },
    'GPT-4.1': {
        'capability': 72.5,
        'turn_level': 38.6,
        'strict': 0.0,
        'color': '#999999',      # Medium gray
        'linestyle': ':',
        'marker': 's',
        'zorder': 1
    },
    'GPT-4.1 Mini': {
        'capability': 71.5,
        'turn_level': 41.6,
        'strict': 1.0,
        'color': '#AAAAAA',      # Light gray
        'linestyle': '-.',
        'marker': '^',
        'zorder': 1
    },
    'Atlas': {
        'capability': 95.9,
        'turn_level': 85.3,
        'strict': 30.9,
        'color': '#2E86AB',      # Professional blue (colorblind-safe)
        'linestyle': '-',
        'marker': 'D',
        'zorder': 3,
        'linewidth': 3.0
    }
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# X-axis categories
x_positions = [0, 1, 2]
x_labels = [
    'Capability\nDemonstration\n(≥1 Tool Success)',
    'Turn-Level\nConsistency\n(Per-Turn Success)',
    'Perfect\nExecution\n(Strict Success)'
]

# Plot each model
for model_name, data in models.items():
    y_values = [data['capability'], data['turn_level'], data['strict']]
    linewidth = data.get('linewidth', 2.0)

    ax.plot(
        x_positions,
        y_values,
        label=model_name,
        color=data['color'],
        linestyle=data['linestyle'],
        marker=data['marker'],
        markersize=8,
        linewidth=linewidth,
        zorder=data['zorder'],
        markerfacecolor=data['color'] if model_name != 'Atlas' else data['color'],
        markeredgewidth=1.5,
        markeredgecolor=data['color']
    )

# Styling
ax.set_xlim(-0.2, 2.2)
ax.set_ylim(-5, 105)
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, ha='center')
ax.set_ylabel('Success Rate (%)', fontweight='bold')
ax.set_title('The Consistency Gap: Baseline Capability vs. Atlas Learning\nProduction-Realistic CRM Agent Evaluation (194 conversations)',
             fontweight='bold', pad=20)

# Grid
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Add horizontal reference lines
ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
ax.axhline(y=75, color='gray', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)

# Legend
legend = ax.legend(
    loc='upper right',
    frameon=True,
    fancybox=False,
    shadow=False,
    framealpha=0.95,
    edgecolor='black',
    facecolor='white'
)
legend.get_frame().set_linewidth(1.0)

# Add annotations for key improvements
# Atlas vs baselines at capability level
ax.annotate(
    '+19.9 pp',
    xy=(0, 95.9),
    xytext=(0.3, 88),
    fontsize=9,
    color='#2E86AB',
    weight='bold',
    ha='left',
    arrowprops=dict(
        arrowstyle='->',
        color='#2E86AB',
        lw=1.5,
        connectionstyle='arc3,rad=0.2'
    )
)

# Atlas strict success improvement
ax.annotate(
    '30.9x vs Mini\n4.4x vs Claude',
    xy=(2, 30.9),
    xytext=(1.5, 45),
    fontsize=9,
    color='#2E86AB',
    weight='bold',
    ha='center',
    arrowprops=dict(
        arrowstyle='->',
        color='#2E86AB',
        lw=1.5,
        connectionstyle='arc3,rad=-0.3'
    )
)

# Baseline clustering annotation
ax.annotate(
    'Baselines cluster:\n71-76% capability',
    xy=(0, 74),
    xytext=(-0.15, 60),
    fontsize=9,
    color='#666666',
    style='italic',
    ha='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#666666', alpha=0.8)
)

# Consistency gap annotation
ax.annotate(
    'Consistency Gap:\nBaselines drop to 0-7%',
    xy=(2, 4),
    xytext=(1.2, 15),
    fontsize=9,
    color='#666666',
    style='italic',
    ha='center',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#666666', alpha=0.8)
)

# Add shaded region showing "dominated" area below Atlas
atlas_capability = models['Atlas']['capability']
atlas_turn = models['Atlas']['turn_level']
atlas_strict = models['Atlas']['strict']

ax.fill_between(
    x_positions,
    [0, 0, 0],
    [atlas_capability, atlas_turn, atlas_strict],
    alpha=0.05,
    color='#2E86AB',
    zorder=0
)

# Add text box with key finding
textstr = 'Key Finding: Models demonstrate capability (71-76%) but lack consistency (0-7%).\nAtlas bridges the gap through continual learning (95.9% → 30.9%).'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.15, edgecolor='black', linewidth=1)
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', bbox=props, style='italic')

# Spine styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# Tight layout
plt.tight_layout()

# Save outputs
output_dir = Path('artifacts/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Save as SVG (vector, for papers)
svg_path = output_dir / 'consistency_gap_bridge.svg'
plt.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"✓ Saved vector graphic: {svg_path}")

# Save as high-res PNG
png_path = output_dir / 'consistency_gap_bridge.png'
plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
print(f"✓ Saved high-res PNG: {png_path}")

# Save as PDF (for LaTeX)
pdf_path = output_dir / 'consistency_gap_bridge.pdf'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"✓ Saved PDF: {pdf_path}")

print(f"\n📊 Diagram 1 generated successfully!")
print(f"   Location: {output_dir}")
print(f"   Formats: SVG (vector), PNG (300 DPI), PDF (LaTeX)")

# Display (if running interactively)
# plt.show()
