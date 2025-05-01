import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Set the figure style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Data from the React component
# Pretraining - w/o Soft Masking
pre_no_soft_epochs = [0, 5, 10, 15]
pre_no_soft_loss = [3.3505, 3.2943, 3.2708, 3.2514]

# Fine-tuning - w/o Soft Masking
ft_no_soft_epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
ft_no_soft_loss = [1.7934, 1.7522, 1.7502, 1.7353, 1.6973, 1.6418, 1.5592, 1.4659, 1.3282, 1.2201, 1.0777, 1.0258, 1.0286, 0.9913, 0.9808]

# Pretraining - w/o Semantic Masking
pre_no_semantic_epochs = [0, 5, 10, 15]
pre_no_semantic_loss = [3.3538, 3.2934, 3.2336, 3.2017]

# Fine-tuning - w/o Semantic Masking
ft_no_semantic_epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
ft_no_semantic_loss = [1.8254, 1.7498, 1.6917, 1.5850, 1.4769, 1.4377, 1.3549, 1.3087, 1.2397, 1.1735, 1.0895, 1.0885, 1.0722, 1.0485, 0.9890]

# Pretraining - GMLM (Full)
pre_gmlm_epochs = [0, 5, 10, 15]
pre_gmlm_loss = [3.1976, 3.1617, 3.1413, 3.1288]

# Fine-tuning - GMLM (Full)
ft_gmlm_epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
ft_gmlm_loss = [1.8163, 1.2394, 0.9770, 0.9546, 0.8836, 0.8985, 0.8877, 0.7988, 0.8453, 0.8359, 0.8057, 0.7863, 0.7687, 0.7933, 0.7615]

# Create a figure with the right dimensions for an academic paper
fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

# Define colors that are distinct and work well in print
colors = {
    'no_soft': '#6A5ACD',       # Slate blue
    'no_semantic': '#2E8B57',   # Sea green
    'gmlm': '#FF7F50'           # Coral
}

# Plot the data
# w/o Soft Masking
ax.plot(pre_no_soft_epochs, pre_no_soft_loss, color=colors['no_soft'], marker='o', 
        linewidth=2, markersize=6, label='Pretraining - w/o Soft Masking')
ax.plot(ft_no_soft_epochs, ft_no_soft_loss, color=colors['no_soft'], marker='o', 
        linewidth=2, markersize=6, linestyle='--', label='Fine-Tuning - w/o Soft Masking')

# w/o Semantic Masking
ax.plot(pre_no_semantic_epochs, pre_no_semantic_loss, color=colors['no_semantic'], marker='s', 
        linewidth=2, markersize=6, label='Pretraining - w/o Semantic Masking')
ax.plot(ft_no_semantic_epochs, ft_no_semantic_loss, color=colors['no_semantic'], marker='s', 
        linewidth=2, markersize=6, linestyle='--', label='Fine-Tuning - w/o Semantic Masking')

# GMLM (Full)
ax.plot(pre_gmlm_epochs, pre_gmlm_loss, color=colors['gmlm'], marker='^', 
        linewidth=2, markersize=6, label='Pretraining - GMLM (Full)')
ax.plot(ft_gmlm_epochs, ft_gmlm_loss, color=colors['gmlm'], marker='^', 
        linewidth=2, markersize=6, linestyle='--', label='Fine-Tuning - GMLM (Full)')

# Set the axis limits similar to the original plot
ax.set_ylim(0.7, 3.5)
ax.set_xlim(-2, 145)

# Add labels and title
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Loss', fontweight='bold')
ax.set_title('GMLM Model Variants Loss Comparison', fontweight='bold', pad=15)

# Customize the grid
ax.grid(True, linestyle='--', alpha=0.7)

# Improve tick spacing
ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

# Add a subtle box around the plot
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# Create a better organized legend
# First create custom legend entries
legend_elements = [
    # Model variants
    Line2D([0], [0], color=colors['no_soft'], lw=2, marker='o', markersize=6, label='w/o Soft Masking'),
    Line2D([0], [0], color=colors['no_semantic'], lw=2, marker='s', markersize=6, label='w/o Semantic Masking'),
    Line2D([0], [0], color=colors['gmlm'], lw=2, marker='^', markersize=6, label='GMLM (Full)'),
    # # Training phases
    Line2D([0], [0], color='gray', lw=2, label='Pretraining'),
    Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Fine-Tuning')
]

# Add the legend with better positioning
legend1 = ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, edgecolor='gray')
legend1.get_title().set_fontweight('bold')
ax.add_artist(legend1)

# Adjust layout and ensure tight fit
plt.tight_layout()

# Save the figure in high resolution for academic publication
plt.savefig('gmlm_loss_comparison.png', dpi=600, bbox_inches='tight')

# Show the plot
plt.show()