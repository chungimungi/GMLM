import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

def generate_semantic_mask(data, mask_ratio):
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    num_train = train_idx.size(0)
    num_mask = max(1, int(mask_ratio * num_train))
    
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    train_degrees = deg[train_idx]
    
    probs = train_degrees / train_degrees.sum()
    sampled = train_idx[torch.multinomial(probs, num_mask, replacement=False)]
    
    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    mask[sampled] = True
    return mask

def soft_masking(x, mask, mask_token, beta=0.7):
    x_masked = x.clone()
    x_masked[mask] = (1 - beta) * x[mask] + beta * mask_token
    return x_masked

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)

    similarity_matrix = torch.mm(representations, representations.t())
    similarity_matrix = similarity_matrix / temperature

    mask = torch.eye(2 * N, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    positive_indices = (torch.arange(2 * N, device=z1.device) + N) % (2 * N)

    loss = F.cross_entropy(similarity_matrix, positive_indices)
    return loss
