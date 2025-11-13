#!/usr/bin/env python3
"""
Generate Diagram 1: Consistency Gap Bridge (Grouped Bar Chart)
Visualization showing baseline capability vs Atlas's bridging of the consistency gap
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Ensure output directory exists
output_dir = Path('artifacts/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Data from reply-case-study.md (194/400 conversations, 48.5% complete)
models = ['Claude 4.5\nSonnet', 'GPT-4.1', 'GPT-4.1\nMini', 'Atlas']
colors = ['#475569', '#94a3b8', '#cbd5e1', '#3b82f6']  # Dark gray, medium gray, light gray, Arc blue

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
        edgecolor = 'white'  # White border for clean look
        linewidth = 2
        alpha = 1.0
    else:
        edgecolor = 'white'
        linewidth = 2
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

        # Match text color to bar color for visual consistency
        text_color = color if model == 'Atlas' else '#334155'  # Use darker gray for text readability

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight=fontweight,
            color=text_color
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
