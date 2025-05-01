import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

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


def soft_masking(x, mask, mask_token, beta=0.7):
    # Get the device of the input tensor
    device = x.device
    mask = mask.to(device)
    mask_token = mask_token.to(device)
    
    x_masked = x.clone()
    x_masked[mask] = (1 - beta) * x[mask] + beta * mask_token
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

