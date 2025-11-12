#!/usr/bin/env python3
"""Generate token efficiency and cost reduction visualization for Reply presentation."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Ensure output directory exists
output_dir = Path(__file__).parent.parent / "artifacts" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

# Data from case study
models = ["GPT-4.1 Mini", "Claude 4.5 Sonnet", "Atlas"]
cost_per_strict_success = [0.46, 0.68, 0.15]
tokens_per_strict_success = [884700, 162600, 96372]
strict_success_rates = [1.0, 7.0, 30.9]

# Color scheme: gradient of grays for baselines, Arc blue for Atlas
colors = ['#cbd5e1', '#94a3b8', '#3b82f6']  # Light gray, medium gray, Arc blue

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Token Efficiency: Cost and Success Rate Analysis',
             fontsize=16, fontweight='bold', y=0.98)

# Left plot: Cost per Strict Success
bars1 = ax1.barh(models, cost_per_strict_success, color=colors, edgecolor='white', linewidth=2)
ax1.set_xlabel('Cost per Strict Success ($)', fontsize=12, fontweight='bold')
ax1.set_title('Production Cost Efficiency', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Add value labels
for i, (bar, cost) in enumerate(zip(bars1, cost_per_strict_success)):
    width = bar.get_width()
    label = f'${cost:.2f}'
    ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
             label, ha='left', va='center', fontweight='bold', fontsize=11)

ax1.set_xlim(0, 0.8)

# Right plot: Tokens per Strict Success (in thousands)
tokens_in_k = [t / 1000 for t in tokens_per_strict_success]
bars2 = ax2.barh(models, tokens_in_k, color=colors, edgecolor='white', linewidth=2)
ax2.set_xlabel('Tokens per Strict Success (thousands)', fontsize=12, fontweight='bold')
ax2.set_title('Token Efficiency', fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Add value labels
for i, (bar, tokens_k) in enumerate(zip(bars2, tokens_in_k)):
    width = bar.get_width()
    label = f'{tokens_k:,.0f}k'
    ax2.text(width + 10, bar.get_y() + bar.get_height()/2,
             label, ha='left', va='center', fontweight='bold', fontsize=11)

ax2.set_xlim(0, 1000)

# Add footer with key insight
fig.text(0.5, 0.02,
         'Continual learning achieves production-grade reliability at significantly lower cost per successful outcome',
         ha='center', fontsize=11, style='italic', color='#475569')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(output_dir / "token_efficiency_cost_reduction.png", dpi=300, bbox_inches='tight')
print(f"Saved visualization to {output_dir / 'token_efficiency_cost_reduction.png'}")

# Create simplified single-panel version for presentation slide
fig2, ax = plt.subplots(figsize=(10, 6))
fig2.suptitle('Cost Efficiency: Strict Success (Perfect Execution)',
             fontsize=16, fontweight='bold', y=0.96)

bars = ax.barh(models, cost_per_strict_success, color=colors, edgecolor='white', linewidth=2)
ax.set_xlabel('Cost per Strict Success ($)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add value labels
for i, (bar, cost) in enumerate(zip(bars, cost_per_strict_success)):
    width = bar.get_width()
    label = f'${cost:.2f}'
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
            label, ha='left', va='center', fontweight='bold', fontsize=12)

ax.set_xlim(0, 0.8)
ax.set_ylim(-0.6, 2.6)

# Add footer
fig2.text(0.5, 0.03,
         'Inference-time continual learning achieves production-grade reliability at significantly lower cost',
         ha='center', fontsize=11, style='italic', color='#475569')

plt.tight_layout(rect=[0, 0.06, 1, 0.94])
plt.savefig(output_dir / "token_efficiency_presentation_slide.png", dpi=300, bbox_inches='tight')
print(f"Saved presentation slide to {output_dir / 'token_efficiency_presentation_slide.png'}")
print(f"\n✓ Token efficiency visualizations generated successfully!")
