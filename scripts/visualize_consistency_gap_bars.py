#!/usr/bin/env python3
"""
Generate Diagram 1: Consistency Gap Bridge (Grouped Bar Chart)
Academic-quality visualization showing baseline capability vs Atlas's bridging of the consistency gap
"""

import matplotlib.pyplot as plt
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
})

# Data from reply-case-study.md (194/400 conversations, 48.5% complete)
models = ['Claude 4.5\nSonnet', 'GPT-4.1', 'GPT-4.1\nMini', 'Atlas']
colors = ['#000000', '#4D4D4D', '#A0A0A0', '#2E86AB']  # Black, Dark Grey, Light Grey, Blue

# Three metrics (groups)
capability_data = [76.0, 72.5, 71.5, 95.9]  # Conversations w/ ≥1 tool success
turn_level_data = [45.8, 38.6, 41.6, 85.3]  # Turn-level tool success
strict_data = [7.0, 0.0, 1.0, 30.9]         # Strict success (all turns)

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Bar settings
bar_width = 0.18
group_spacing = 1.0  # Professional spacing between groups
x_positions = np.arange(3) * group_spacing  # 3 groups

# Create bars for each model
for i, (model, color) in enumerate(zip(models, colors)):
    offset = (i - 1.5) * bar_width  # Center the group

    # Get data for this model
    heights = [capability_data[i], turn_level_data[i], strict_data[i]]

    # Determine bar properties
    if model == 'Atlas':
        edgecolor = '#1a5a7a'  # Darker blue for subtle border
        linewidth = 1.5
        alpha = 1.0
    else:
        edgecolor = '#333333'
        linewidth = 0.5
        alpha = 1.0

    bars = ax.bar(
        x_positions + offset,
        heights,
        bar_width,
        label=model,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=3 if model == 'Atlas' else 2
    )

    # Add value labels on top of bars
    for bar, height in zip(bars, heights):
        fontweight = 'bold' if model == 'Atlas' else 'normal'
        fontsize = 10 if model == 'Atlas' else 9

        if height > 0:  # Bar is visible
            label_y = height + 2
        else:  # Bar has zero height - show label at baseline
            label_y = 2

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight=fontweight,
            color=color if model == 'Atlas' else '#333333'
        )

# Styling
ax.set_ylim(0, 108)
ax.set_xlim(-0.5, 2.5)
ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=13)
ax.set_title('CRM Agent Evaluation of Inference-Time Continual Learning',
             fontweight='bold', pad=20, fontsize=15)

# X-axis labels
group_labels = [
    'Conversational\nSuccess\n(≥1 Tool Success)',
    'Turn-Level\nConsistency\n(Per-Turn Success)',
    'Perfect\nExecution\n(Strict Success)'
]
ax.set_xticks(x_positions)
ax.set_xticklabels(group_labels, ha='center', fontsize=11)

# Grid
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
ax.set_axisbelow(True)

# Add horizontal reference lines
for y in [25, 50, 75, 100]:
    ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.3, alpha=0.2, zorder=0)

# Legend
legend = ax.legend(
    loc='upper right',
    frameon=True,
    fancybox=False,
    shadow=False,
    framealpha=0.95,
    edgecolor='black',
    facecolor='white',
    ncol=1
)
legend.get_frame().set_linewidth(1.0)

# Annotations removed per user request - arrows interfere with bars

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
svg_path = output_dir / 'consistency_gap_bridge_bars.svg'
plt.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"✓ Saved vector graphic: {svg_path}")

# Save as high-res PNG
png_path = output_dir / 'consistency_gap_bridge_bars.png'
plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
print(f"✓ Saved high-res PNG: {png_path}")

# Save as PDF (for LaTeX)
pdf_path = output_dir / 'consistency_gap_bridge_bars.pdf'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"✓ Saved PDF: {pdf_path}")

print(f"\n📊 Diagram 1 (Grouped Bar Chart) generated successfully!")
print(f"   Location: {output_dir}")
print(f"   Formats: SVG (vector), PNG (300 DPI), PDF (LaTeX)")
