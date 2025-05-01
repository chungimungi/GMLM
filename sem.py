import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torchinfo
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
import torch_geometric.transforms as T
from transformers import AutoModel, AutoTokenizer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#----------------------------------------
# Utility Functions
#----------------------------------------

def generate_semantic_mask(data, mask_ratio, split_edge=None):
    # For link prediction datasets where train_mask doesn't exist
    if not hasattr(data, 'train_mask'):
        if split_edge is None:
            # If no split_edge is provided, create mask from all nodes
            train_nodes = torch.arange(data.num_nodes, device=data.x.device)
        else:
            # Create mask from nodes that appear in training edges
            train_edges = split_edge['train']['edge']
            train_nodes = torch.unique(train_edges.flatten())
        
        num_train = train_nodes.size(0)
        num_mask = max(1, int(mask_ratio * num_train))
        
        # Get degrees of all nodes
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        
        # Get degrees of training nodes
        train_degrees = deg[train_nodes]
        
        # Sample nodes based on their degrees
        probs = train_degrees / train_degrees.sum()
        sampled = train_nodes[torch.multinomial(probs, num_mask, replacement=False)]
    else:
        # Original code for node classification datasets
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        num_train = train_idx.size(0)
        num_mask = max(1, int(mask_ratio * num_train))
        
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        train_degrees = deg[train_idx]
        
        probs = train_degrees / train_degrees.sum()
        sampled = train_idx[torch.multinomial(probs, num_mask, replacement=False)]
    
    # Create mask
    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    mask[sampled] = True
    return mask


def hard_masking(x, mask, mask_token):
    """
    Apply hard masking to node features instead of soft masking.
    Completely replaces masked node features with mask token.
    """
    # Get the device of the input tensor
    device = x.device
    mask = mask.to(device)
    mask_token = mask_token.to(device)
    
    x_masked = x.clone()
    x_masked[mask] = mask_token
    return x_masked


def nt_xent_loss(z1, z2, temperature=0.5, batch_size=16):
    """
    Compute NT-Xent loss in batches to reduce memory consumption.
    
    Args:
        z1, z2: Embeddings to compare, shape [N, dim]
        temperature: Temperature parameter
        batch_size: Maximum number of samples to process at once
    """
    device = z1.device
    total_loss = 0.0
    num_batches = 0
    total_samples = z1.size(0)
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch_size_curr = end_idx - i
        
        if batch_size_curr <= 1:
            continue
            
        batch_z1 = F.normalize(z1[i:end_idx], dim=1)
        batch_z2 = F.normalize(z2[i:end_idx], dim=1)
        
        batch_emb = torch.cat([batch_z1, batch_z2], dim=0) 
        
        similarity = torch.mm(batch_emb, batch_emb.t()) / temperature
        
        mask = torch.eye(2 * batch_size_curr, dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(mask, -9e15)
        
        pos_indices = (torch.arange(2 * batch_size_curr, device=device) + batch_size_curr) % (2 * batch_size_curr)
        
        batch_loss = F.cross_entropy(similarity, pos_indices)
        
        total_loss += batch_loss
        num_batches += 1
    
    # Return average loss or zero if no valid batches
    if num_batches == 0:
        return torch.tensor(0.0, device=device)
    
    return total_loss / num_batches

#----------------------------------------
# Model Definition
#----------------------------------------

class GraphMaskedLM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, nhead=4, model_name='distilbert/distilbert-base-uncased'):
        super(GraphMaskedLM, self).__init__()
        # Initialize mask token as a learnable parameter
        self.mask_token = nn.Parameter(torch.zeros(in_channels))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)  # Better initialization for mask token
        
        self.gat1 = GATConv(in_channels, hidden_channels, heads=nhead, dropout=0.6)
        self.bn1 = nn.BatchNorm1d(hidden_channels * nhead)
        self.dropout1 = nn.Dropout(0.3)
        
        self.gat2 = GATConv(hidden_channels * nhead, hidden_channels, heads=2, concat=True, dropout=0.6)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        bert_hidden_size = self.encoder.config.hidden_size  
        self.proj = nn.Linear(hidden_channels * 2, bert_hidden_size)
        
        fusion_hidden = 512
        self.fusion_network = nn.Sequential(
            nn.Linear(bert_hidden_size * 2, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, num_classes)
        )

    def get_graph_embeddings(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout1(x)
        
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout2(x)
        
        x = self.proj(x)
        return x

    def forward(self, x, edge_index, mask):
        x_graph = self.get_graph_embeddings(x, edge_index)
        
        texts = ["[MASK]" if flag else "node" for flag in mask.cpu().tolist()]
        encodings = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        ).to(x.device)
        
        with torch.set_grad_enabled(self.training):
            bert_output = self.encoder(**encodings)
            bert_embeddings = bert_output.last_hidden_state[:, 0, :] 
        
        combined = torch.cat([x_graph, bert_embeddings], dim=1)
        fused = self.fusion_network(combined)
        
        logits = self.classifier(fused)
        return logits

#----------------------------------------
# Training Functions
#----------------------------------------

def pretrain_contrastive(model, data, pretrain_epochs=20, temp=0.5):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    
    best_loss = float('inf')

    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        
        # Generate two different masks for creating two views
        mask1 = generate_semantic_mask(data, random.uniform(0.2, 0.4))
        mask2 = generate_semantic_mask(data, random.uniform(0.2, 0.4))
        
        # Apply hard masking instead of soft masking
        x1 = hard_masking(data.x, mask1, model.mask_token)
        x2 = hard_masking(data.x, mask2, model.mask_token)
        
        g1 = model.get_graph_embeddings(x1, data.edge_index)
        g2 = model.get_graph_embeddings(x2, data.edge_index)
        
        loss = nt_xent_loss(g1, g2, temperature=temp)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"[Pretrain] Epoch: {epoch}, Loss: {loss.item():.4f}")

def train_model(model, data, num_epochs=150):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Generate masks for semantic nodes
        mask = generate_semantic_mask(data, random.uniform(0.2, 0.4))
        
        # Apply hard masking instead of soft masking
        x_masked = hard_masking(data.x, mask, model.mask_token)
        
        logits = model(x_masked, data.edge_index, mask)
        loss = criterion(logits[mask], data.y[mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"[Train] Epoch: {epoch}, Loss: {loss.item():.4f}")

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        # Create empty mask (no masking during evaluation)
        mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        logits = model(data.x, data.edge_index, mask)
        
        # Calculate accuracy for train/val/test sets if available
        if hasattr(data, 'train_mask'):
            train_acc = accuracy(logits[data.train_mask], data.y[data.train_mask])
            val_acc = accuracy(logits[data.val_mask], data.y[data.val_mask])
            test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
            print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        else:
            print("No train/val/test masks available for evaluation")

def accuracy(pred, target):
    pred = pred.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return correct / len(target)

#----------------------------------------
# Main Execution
#----------------------------------------

def main():
    # Dataset configurations
    dataset_configs = [
        # {'name': 'Amazon-Photo', 'class': Amazon, 'kwargs': {'root': './data/Amazon-Photo', 'name': 'Photo'}, 'split_transform': True},
        # {'name': 'Amazon-Computers', 'class': Amazon, 'kwargs': {'root': './data/Amazon-Computers', 'name': 'Computers'}, 'split_transform': True},
        # {'name': 'Coauthor-CS', 'class': Coauthor, 'kwargs': {'root': './data/Coauthor-CS', 'name': 'CS'}, 'split_transform': True},
        # {'name': 'Cora', 'class': Planetoid, 'kwargs': {'root': './data', 'name': 'Cora'}, 'split_transform': True},
        {'name': 'Citeseer', 'class': Planetoid, 'kwargs': {'root': './data', 'name': 'Citeseer'}, 'split_transform': True},
        # {'name': 'PubMed', 'class': Planetoid, 'kwargs': {'root': './data', 'name': 'PubMed'}, 'split_transform': True},
    ]

    for config in dataset_configs:
        dataset_name = config['name']
        dataset_class = config['class']
        kwargs = config['kwargs']
        split_transform = config['split_transform']

        print(f"\n=== Running on {dataset_name} dataset ===")

        # Apply transforms
        transforms = [T.NormalizeFeatures()]
        if split_transform:
            transforms.append(T.RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2))

        # Load dataset
        dataset = dataset_class(transform=T.Compose(transforms), **kwargs)
        data = dataset[0].to(device)

        # Initialize model
        in_channels = dataset.num_features
        num_classes = dataset.num_classes
        model = GraphMaskedLM(in_channels, hidden_channels=128, num_classes=num_classes, nhead=4).to(device)
        print(torchinfo.summary(model))

        # Contrastive pretraining
        print("Starting contrastive pretraining...")
        pretrain_contrastive(model, data, pretrain_epochs=20, temp=0.5)

        # Fine-tuning
        print("Starting fine-tuning...")
        train_model(model, data, num_epochs=150)

        # Evaluate and save model
        # print("Evaluating model...")
        # evaluate_model(model, data)
        torch.save(model.state_dict(), f'{dataset_name.lower()}_model_semantic.pt')

    print("\n=== All datasets processed successfully! ===")

if __name__ == "__main__":
    main()