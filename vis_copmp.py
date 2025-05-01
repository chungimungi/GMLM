import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from gmlm.models import GraphMaskedLM
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

# Define model paths and labels
model_configs = [
    {'path': 'citeseer_model_semantic.pt', 'label': 'w/o Soft Masking'},
    {'path': 'citeseer_model_soft.pt', 'label': 'w/o Semantic Masking'},
    {'path': 'citeseer_model.pt', 'label': 'GMLM(Full)'}
]

# Initialize model architecture
print("Initializing models...")
in_channels = dataset.num_features
num_classes = dataset.num_classes
hidden_channels = 128
nhead = 4

# Load all three models
models = []
embeddings = []
loaded_models = []

for config in model_configs:
    model_path = config['path']
    model_label = config['label']
    
    # Initialize model with the same architecture
    model = GraphMaskedLM(in_channels, hidden_channels=hidden_channels, num_classes=num_classes, nhead=nhead).to(device)
    
    # Try to load the pretrained model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model '{model_label}' from {model_path}")
        
        # Get embeddings
        print(f"Extracting embeddings from '{model_label}'...")
        model_embeddings = get_embeddings(model, data)
        
        models.append(model)
        embeddings.append(model_embeddings)
        loaded_models.append(model_label)
    except FileNotFoundError:
        print(f"Warning: Model file not found at {model_path}")

# Check if we have at least one model loaded
if not loaded_models:
    print("Error: No models could be loaded. Please check the model paths.")
    exit()

print(f"Successfully loaded {len(loaded_models)} models: {', '.join(loaded_models)}")

# Define visualization methods for multiple models
def visualize_tsne_multiple(embeddings_list, model_labels, node_labels, perplexity=30, n_iter=1000):
    print(f"Performing t-SNE with perplexity={perplexity}...")
    
    num_models = len(embeddings_list)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 8))
    fig.suptitle(f't-SNE Visualization of Citeseer Embeddings (perplexity={perplexity})', fontsize=20)
    
    if num_models == 1:
        axes = [axes]  # Make axes iterable when there's only one subplot
    
    # Create a colormap
    num_classes = len(np.unique(node_labels))
    colors = sns.color_palette('tab10', n_colors=num_classes)
    cmap = ListedColormap(colors)
    
    # Process each model
    for i, (embeddings, label, ax) in enumerate(zip(embeddings_list, model_labels, axes)):
        # t-SNE for model embeddings
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        reduced = tsne.fit_transform(embeddings)
        
        # Plot t-SNE
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                           c=node_labels, cmap=cmap, alpha=0.99, s=40)
        ax.set_title(label, fontsize=18)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=16)
        
        if i == 0:  # Only add y-label to the first plot
            ax.set_ylabel('t-SNE Dimension 2', fontsize=16)
    
    # Add legend to the last plot
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=colors[i], label=label_names[i],
                               markersize=10, markeredgecolor='k') for i in range(num_classes)]
    axes[-1].legend(handles=legend_elements, loc='best', fontsize=14, bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f'visualization_output/citeseer_tsne_comparison_perp{perplexity}.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"t-SNE comparison saved to {filename}")

def visualize_umap_multiple(embeddings_list, model_labels, node_labels, n_neighbors=15, min_dist=0.1):
    print(f"Performing UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    
    num_models = len(embeddings_list)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 8))
    fig.suptitle(f'UMAP Visualization of Citeseer Embeddings (n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=20)
    
    if num_models == 1:
        axes = [axes]  # Make axes iterable when there's only one subplot
    
    # Create a colormap
    num_classes = len(np.unique(node_labels))
    colors = sns.color_palette('tab10', n_colors=num_classes)
    cmap = ListedColormap(colors)
    
    # Process each model
    for i, (embeddings, label, ax) in enumerate(zip(embeddings_list, model_labels, axes)):
        # UMAP for model embeddings
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        
        # Plot UMAP
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                           c=node_labels, cmap=cmap, alpha=0.99, s=40)
        ax.set_title(label, fontsize=18)
        ax.set_xlabel('UMAP Dimension 1', fontsize=16)
        
        if i == 0:  # Only add y-label to the first plot
            ax.set_ylabel('UMAP Dimension 2', fontsize=16)
    
    # Add legend to the last plot
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=colors[i], label=label_names[i],
                               markersize=10, markeredgecolor='k') for i in range(num_classes)]
    axes[-1].legend(handles=legend_elements, loc='best', fontsize=14, bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f'visualization_output/citeseer_umap_comparison_n{n_neighbors}_d{min_dist}.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"UMAP comparison saved to {filename}")

# Generate visualizations
print("Generating comparison visualizations...")

# t-SNE with different perplexity values
for perplexity in [30]:
    visualize_tsne_multiple(embeddings, loaded_models, labels, perplexity=perplexity)

# UMAP with different parameters
for n_neighbors in [15]:
    for min_dist in [0.1]:
        visualize_umap_multiple(embeddings, loaded_models, labels, n_neighbors=n_neighbors, min_dist=min_dist)

print("All visualizations completed successfully!")