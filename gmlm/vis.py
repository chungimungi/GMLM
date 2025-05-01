import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from models import GraphMaskedLM
from umap import UMAP
import seaborn as sns
import os
from matplotlib.colors import ListedColormap

plt.style.use('bmh')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 14,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 600,
    'savefig.dpi': 600
})

# Create output directory if it doesn't exist
os.makedirs('visualization_output', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Citeseer dataset
print("Loading Citeseer dataset...")
transforms = [T.NormalizeFeatures(), T.RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)]
dataset = Planetoid(root='./data', name='Citeseer', transform=T.Compose(transforms))
data = dataset[0].to(device)

# Define Citeseer class names
label_names = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']  # Citeseer class names

# Extract node labels
labels = data.y.cpu().numpy()

# Function to get embeddings from a model
def get_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        # Get the graph embeddings without masking
        embeddings = model.get_graph_embeddings(data.x, data.edge_index)
        return embeddings.cpu().numpy()

# Initialize untrained model
print("Initializing untrained model...")
in_channels = dataset.num_features
num_classes = dataset.num_classes
untrained_model = GraphMaskedLM(in_channels, hidden_channels=128, num_classes=num_classes, nhead=4).to(device)

# Get embeddings from untrained model
print("Extracting embeddings from untrained model...")
untrained_embeddings = get_embeddings(untrained_model, data)

# Initialize trained model
print("Initializing trained model...")
trained_model = GraphMaskedLM(in_channels, hidden_channels=128, num_classes=num_classes, nhead=4).to(device)

# Load pre-trained model if available
model_path = 'citeseer_model.pt'
try:
    trained_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded pre-trained model from {model_path}")
    # Get embeddings from trained model
    print("Extracting embeddings from trained model...")
    trained_embeddings = get_embeddings(trained_model, data)
    has_trained_model = True
except FileNotFoundError:
    print(f"No pre-trained model found at {model_path}. Will only visualize untrained embeddings.")
    has_trained_model = False

# Define visualization methods
def visualize_tsne(untrained_embeddings, trained_embeddings=None, labels=None, perplexity=30, n_iter=1000):
    print(f"Performing t-SNE with perplexity={perplexity}...")
    
    # Create figure with proper size and title
    if trained_embeddings is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f't-SNE Visualization of Citeseer Embeddings (perplexity={perplexity})', fontsize=20)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))
        fig.suptitle(f't-SNE Visualization of Untrained Citeseer Embeddings (perplexity={perplexity})', fontsize=20)
    
    # Create a colormap
    num_classes = len(np.unique(labels))
    colors = sns.color_palette('tab10', n_colors=num_classes)
    cmap = ListedColormap(colors)
    
    # t-SNE for untrained embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    untrained_reduced = tsne.fit_transform(untrained_embeddings)
    
    # Plot untrained t-SNE
    scatter1 = ax1.scatter(untrained_reduced[:, 0], untrained_reduced[:, 1], 
                             c=labels, cmap=cmap, alpha=0.99, s=40)
    ax1.set_title('Untrained Model', fontsize=18)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=16)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=16)
    
    # Plot trained t-SNE if available
    if trained_embeddings is not None:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        trained_reduced = tsne.fit_transform(trained_embeddings)
        
        scatter2 = ax2.scatter(trained_reduced[:, 0], trained_reduced[:, 1], 
                               c=labels, cmap=cmap, alpha=0.99, s=40)
        ax2.set_title('Fine-tuned Model', fontsize=18)
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=16)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=16)
    
    # Add legend (common for both plots)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=colors[i], label=label_names[i],
                                   markersize=10, markeredgecolor='k') for i in range(num_classes)]
    if trained_embeddings is not None:
        ax2.legend(handles=legend_elements, loc='best', fontsize=14)
    else:
        ax1.legend(handles=legend_elements, loc='best', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f'visualization_output/citeseer_tsne_comparison_perp{perplexity}.png'
    plt.savefig(filename, dpi=600)
    plt.close()
    print(f"t-SNE comparison saved to {filename}")

def visualize_umap(untrained_embeddings, trained_embeddings=None, labels=None, n_neighbors=15, min_dist=0.1):
    print(f"Performing UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    
    # Create figure with proper size and title
    if trained_embeddings is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'UMAP Visualization of Citeseer Embeddings (n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=20)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))
        fig.suptitle(f'UMAP Visualization of Untrained Citeseer Embeddings (n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=20)
    
    # Create a colormap
    num_classes = len(np.unique(labels))
    colors = sns.color_palette('tab10', n_colors=num_classes)
    cmap = ListedColormap(colors)
    
    # UMAP for untrained embeddings
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    untrained_reduced = reducer.fit_transform(untrained_embeddings)
    
    # Plot untrained UMAP
    scatter1 = ax1.scatter(untrained_reduced[:, 0], untrained_reduced[:, 1], 
                             c=labels, cmap=cmap, alpha=0.99, s=40)
    ax1.set_title('Untrained Model', fontsize=18)
    ax1.set_xlabel('UMAP Dimension 1', fontsize=16)
    ax1.set_ylabel('UMAP Dimension 2', fontsize=16)
    
    # Plot trained UMAP if available
    if trained_embeddings is not None:
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        trained_reduced = reducer.fit_transform(trained_embeddings)
        
        scatter2 = ax2.scatter(trained_reduced[:, 0], trained_reduced[:, 1], 
                               c=labels, cmap=cmap, alpha=0.99, s=40)
        ax2.set_title('Fine-tuned Model', fontsize=18)
        ax2.set_xlabel('UMAP Dimension 1', fontsize=16)
        ax2.set_ylabel('UMAP Dimension 2', fontsize=16)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=colors[i], label=label_names[i],
                                   markersize=10, markeredgecolor='k') for i in range(num_classes)]
    if trained_embeddings is not None:
        ax2.legend(handles=legend_elements, loc='best', fontsize=14)
    else:
        ax1.legend(handles=legend_elements, loc='best', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f'visualization_output/citeseer_umap_comparison_n{n_neighbors}_d{min_dist}.png'
    plt.savefig(filename, dpi=600)
    plt.close()
    print(f"UMAP comparison saved to {filename}")

# Generate multiple visualizations with different parameters
print("Generating comparison visualizations...")

# t-SNE with different perplexity values
for perplexity in [30]:
    if has_trained_model:
        visualize_tsne(untrained_embeddings, trained_embeddings, labels, perplexity=perplexity)
    else:
        visualize_tsne(untrained_embeddings, None, labels, perplexity=perplexity)

# UMAP with different parameters
for n_neighbors in [15]:
    for min_dist in [0.1]:
        if has_trained_model:
            visualize_umap(untrained_embeddings, trained_embeddings, labels, n_neighbors=n_neighbors, min_dist=min_dist)
        else:
            visualize_umap(untrained_embeddings, None, labels, n_neighbors=n_neighbors, min_dist=min_dist)


print("All visualizations completed successfully!")
