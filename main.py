import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.utils import degree
from transformers import AutoModel, AutoTokenizer
import random
import numpy as np
import itertools
import os
import json
from datetime import datetime
from sklearn.metrics import f1_score
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_semantic_mask(data, mask_ratio, split_edge=None):
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        if split_edge is None:
            train_nodes = torch.arange(data.num_nodes, device=data.x.device)
        else:
            train_edges = split_edge['train']['edge']
            train_nodes = torch.unique(train_edges.flatten())

        num_train = train_nodes.size(0)
        num_mask = max(1, min(int(mask_ratio * num_train), num_train))

        edge_index = data.edge_index.to(torch.long)
        deg = degree(edge_index[0], num_nodes=data.num_nodes)
        train_degrees = deg[train_nodes]

        if train_degrees.sum() == 0:
            if num_train > 0:
                sampled_indices = torch.randperm(num_train, device=data.x.device)[:num_mask]
                sampled = train_nodes[sampled_indices]
            else:
                sampled = torch.tensor([], dtype=torch.long, device=data.x.device)
        else:
            probs = train_degrees.float() / train_degrees.sum().float()
            probs = torch.nan_to_num(probs, nan=1.0/num_train if num_train > 0 else 0.0)
            if probs.sum() == 0:
                 sampled = torch.tensor([], dtype=torch.long, device=data.x.device)
            elif num_mask > 0 and num_train > 0:
                 sampled = train_nodes[torch.multinomial(probs, num_mask, replacement=False)]
            else:
                 sampled = torch.tensor([], dtype=torch.long, device=data.x.device)
    else:
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        num_train = train_idx.size(0)
        num_mask = max(1, min(int(mask_ratio * num_train), num_train))

        if num_train == 0 or num_mask == 0:
             mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
             return mask

        edge_index = data.edge_index.to(torch.long)
        deg = degree(edge_index[0], num_nodes=data.num_nodes)
        train_degrees = deg[train_idx]

        if train_degrees.sum() == 0:
            sampled_indices = torch.randperm(num_train, device=data.x.device)[:num_mask]
            sampled = train_idx[sampled_indices]
        else:
            probs = train_degrees.float() / train_degrees.sum().float()
            probs = torch.nan_to_num(probs, nan=1.0/num_train)
            if probs.sum() > 0:
                 sampled = train_idx[torch.multinomial(probs, num_mask, replacement=False)]
            else:
                 sampled_indices = torch.randperm(num_train, device=data.x.device)[:num_mask]
                 sampled = train_idx[sampled_indices]

    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    if sampled.numel() > 0:
        mask[sampled] = True
    return mask


def soft_masking(x, mask, mask_token, beta=0.7):
    device = x.device
    mask = mask.to(device)
    mask_token = mask_token.to(device)
    x_masked = x.clone()
    if mask.any():
        x_masked[mask] = (1 - beta) * x[mask] + beta * mask_token
    return x_masked


def nt_xent_loss(z1, z2, temperature=0.5, batch_size=None):
    device = z1.device
    total_loss = 0.0
    num_batches = 0
    total_samples = z1.size(0)

    if total_samples == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    effective_batch_size = batch_size if batch_size is not None else total_samples

    for i in range(0, total_samples, effective_batch_size):
        end_idx = min(i + effective_batch_size, total_samples)
        batch_size_curr = end_idx - i

        if batch_size_curr <= 1:
            continue

        batch_z1 = F.normalize(z1[i:end_idx], dim=1)
        batch_z2 = F.normalize(z2[i:end_idx], dim=1)

        batch_emb = torch.cat([batch_z1, batch_z2], dim=0)
        similarity = torch.mm(batch_emb, batch_emb.t()) / temperature
        mask = torch.eye(2 * batch_size_curr, dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(mask, -float('inf'))
        pos_indices = torch.arange(batch_size_curr, device=device)
        labels = torch.cat([pos_indices + batch_size_curr, pos_indices], dim=0)
        batch_loss = F.cross_entropy(similarity, labels)
        total_loss += batch_loss * (batch_size_curr / total_samples)
        num_batches += 1

    if num_batches == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss


class GraphMaskedLM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, nhead=4, model_name='distilbert/distilbert-base-uncased'):
        super(GraphMaskedLM, self).__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels))
        self.gat1 = GATConv(in_channels, hidden_channels, heads=nhead, dropout=0.6)
        self.bn1 = nn.BatchNorm1d(hidden_channels * nhead)
        self.dropout1 = nn.Dropout(0.3)
        self.gat2 = GATConv(hidden_channels * nhead, hidden_channels, heads=2, concat=True, dropout=0.6)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.dropout2 = nn.Dropout(0.3)

        print(f"Loading HuggingFace model: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("HuggingFace model loaded.")

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
        edge_index = edge_index.to(torch.long)
        x = self.gat1(x, edge_index)
        if x.size(0) > 1: x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout1(x)
        x = self.gat2(x, edge_index)
        if x.size(0) > 1: x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout2(x)
        x = self.proj(x)
        return x

    def forward(self, x, edge_index, mask):
        edge_index = edge_index.to(torch.long)
        x_graph = self.get_graph_embeddings(x, edge_index)
        texts = ["[MASK]" if m else "node feature" for m in mask.cpu().tolist()]

        if not texts:
             bert_embeddings = torch.zeros(0, self.encoder.config.hidden_size, device=x.device)
        else:
            encodings = self.tokenizer.batch_encode_plus(
                texts, padding=True, truncation=True, max_length=32, return_tensors="pt"
            ).to(x.device)
            with torch.set_grad_enabled(self.training):
                 bert_output = self.encoder(**encodings)
                 bert_embeddings = bert_output.last_hidden_state[:, 0, :]

        if x_graph.size(0) != bert_embeddings.size(0):
             # Try to handle potential size mismatch during evaluation if mask changes size
             if not self.training and bert_embeddings.size(0) == mask.sum().item():
                 # If evaluating and BERT embedding count matches mask count, align based on mask
                 aligned_bert_embeddings = torch.zeros_like(x_graph)
                 aligned_bert_embeddings[mask] = bert_embeddings
                 bert_embeddings = aligned_bert_embeddings
             else:
                 raise ValueError(f"Mismatch in embedding sizes: Graph ({x_graph.size(0)}) vs BERT ({bert_embeddings.size(0)})")


        combined = torch.cat([x_graph, bert_embeddings], dim=1)
        fused = self.fusion_network(combined)
        logits = self.classifier(fused)
        return logits


def setup_optimizer(model, lr_graph, lr_bert, lr_other, weight_decay):
    graph_params = []
    bert_params = []
    other_params = []
    graph_param_names = ['gat1', 'gat2', 'bn1', 'bn2']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('encoder.'):
            bert_params.append(param)
        elif any(name.startswith(p_name + '.') for p_name in graph_param_names):
            graph_params.append(param)
        else:
            other_params.append(param)

    print(f"Optimizer setup: Graph params={len(graph_params)}, BERT params={len(bert_params)}, Other params={len(other_params)}")
    optimizer = optim.AdamW([
        {'params': graph_params, 'lr': lr_graph, 'weight_decay': weight_decay},
        {'params': bert_params, 'lr': lr_bert, 'weight_decay': 0.0},
        {'params': other_params, 'lr': lr_other, 'weight_decay': weight_decay}
    ])
    return optimizer


def pretrain_contrastive(model, data, pretrain_epochs=20, temp=0.5, mask_min=0.2, mask_max=0.4, beta=0.7,
                         lr_graph=1e-3, lr_bert=1e-5, lr_other=1e-4, weight_decay=0.01, verbose=True):
    model.train()
    optimizer = setup_optimizer(model, lr_graph, lr_bert, lr_other, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    losses = []

    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        mask1 = generate_semantic_mask(data, random.uniform(mask_min, mask_max))
        mask2 = generate_semantic_mask(data, random.uniform(mask_min, mask_max))
        x1 = soft_masking(data.x, mask1, model.mask_token, beta=beta)
        x2 = soft_masking(data.x, mask2, model.mask_token, beta=beta)
        g1 = model.get_graph_embeddings(x1, data.edge_index)
        g2 = model.get_graph_embeddings(x2, data.edge_index)
        loss = nt_xent_loss(g1, g2, temperature=temp, batch_size=None)

        if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
             loss.backward()
             optimizer.step()
             scheduler.step()
             losses.append(loss.item())
        else:
             print(f"[Pretrain Warning] Invalid loss encountered (Epoch {epoch}): {loss}. Skipping step.")
             losses.append(np.nan)

        if verbose and (epoch % 5 == 0 or epoch == pretrain_epochs - 1):
            current_lr_g = optimizer.param_groups[0]['lr']
            current_lr_b = optimizer.param_groups[1]['lr']
            current_lr_o = optimizer.param_groups[2]['lr']
            print(f"[Pretrain] Epoch: {epoch}, Loss: {loss.item():.4f}, LRs: G={current_lr_g:.1e}, B={current_lr_b:.1e}, O={current_lr_o:.1e}")

    avg_loss = np.nanmean(losses[-5:]) if losses else 0
    if verbose:
        print(f"[Pretrain] Final Avg Loss (last 5 epochs): {avg_loss:.4f}")
    return avg_loss


def train_model(model, data, num_epochs=150, mask_min=0.2, mask_max=0.4, beta=0.7,
                lr_graph=1e-3, lr_bert=1e-5, lr_other=1e-4, weight_decay=0.01, patience=20, verbose=True):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = setup_optimizer(model, lr_graph, lr_bert, lr_other, weight_decay)

    losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_dict = None
    stopped_epoch = num_epochs

    use_early_stopping = hasattr(data, 'val_mask') and data.val_mask is not None and data.val_mask.any()
    if not use_early_stopping:
        print("[Train Warning] No valid validation mask found. Early stopping disabled.")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        mask = generate_semantic_mask(data, random.uniform(mask_min, mask_max))
        if not mask.any():
             print(f"[Train Warning] Epoch {epoch}: No nodes selected by mask for training. Skipping epoch.")
             continue

        x_soft = soft_masking(data.x, mask, model.mask_token, beta=beta)
        logits = model(x_soft, data.edge_index, mask)

        if data.y[mask].numel() == 0:
            print(f"[Train Warning] Epoch {epoch}: No labels available for the selected mask. Skipping loss computation.")
            continue

        loss = criterion(logits[mask], data.y[mask])

        if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        else:
            print(f"[Train Warning] Invalid training loss encountered (Epoch {epoch}): {loss}. Skipping step.")
            losses.append(np.nan)
            continue

        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
             current_lr_g = optimizer.param_groups[0]['lr']
             current_lr_b = optimizer.param_groups[1]['lr']
             current_lr_o = optimizer.param_groups[2]['lr']
             print(f"[Train] Epoch: {epoch}, Loss: {loss.item():.4f}, LRs: G={current_lr_g:.1e}, B={current_lr_b:.1e}, O={current_lr_o:.1e}")

        if use_early_stopping:
            model.eval()
            with torch.no_grad():
                val_mask = data.val_mask
                # Use original features for validation, pass val_mask to forward
                val_logits = model(data.x, data.edge_index, val_mask)
                val_loss = eval_criterion(val_logits[val_mask], data.y[val_mask])

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"[Validate] Epoch: {epoch}, Val Loss: {val_loss.item():.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state_dict = copy.deepcopy(model.state_dict())
                if verbose:
                    print(f"    New best validation loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} after {patience} epochs with no improvement.")
                stopped_epoch = epoch
                break
        else:
            # If not using early stopping, consider saving the last model state?
            # Or save based on training loss? For now, just runs full epochs.
            pass


    avg_loss = np.nanmean(losses[-5:]) if losses else 0
    if verbose:
        print(f"[Train] Final Avg Training Loss (last 5 epochs): {avg_loss:.4f}")

    if use_early_stopping and best_model_state_dict is not None:
        print(f"Loading best model state from epoch {stopped_epoch - patience} with Val Loss: {best_val_loss:.4f}")
        model.load_state_dict(best_model_state_dict)
    elif not use_early_stopping:
         print("Training completed full epochs without early stopping.")
    else:
         print("Warning: Early stopping enabled but no best model state saved.")


    return avg_loss


def evaluate_model(model, data):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    if not hasattr(data, 'test_mask') or data.test_mask is None or not data.test_mask.any():
        print("[Evaluate Error] No valid test_mask found in data. Cannot evaluate.")
        return {'loss': float('nan'), 'accuracy': 0.0, 'f1': 0.0}

    with torch.no_grad():
        test_mask = data.test_mask
        # Use original features for evaluation, pass test_mask to forward
        logits = model(data.x, data.edge_index, test_mask)
        loss = criterion(logits[test_mask], data.y[test_mask])
        pred = logits[test_mask].max(dim=1)[1]
        acc = pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
        f1 = calculate_f1(pred.cpu().numpy(), data.y[test_mask].cpu().numpy())
        print(f"Test Loss: {loss.item():.4f}, Accuracy: {acc*100:.2f}%, F1: {f1:.4f}")
        return {
            'loss': loss.item(),
            'accuracy': acc * 100,
            'f1': f1
        }


def calculate_f1(y_pred, y_true):
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    beta = 0.7
    mask_min = 0.2
    mask_max = 0.4
    temp = 0.5
    pretrain_epochs = 20
    train_epochs = 150
    patience = 20
    lr_graph = 1e-3
    lr_bert = 1e-5
    lr_other = 1e-4
    weight_decay = 0.01
    hidden_channels = 128
    nhead = 4

    dataset_configs = [
        {'name': 'Amazon-Photo', 'class': Amazon, 'kwargs': {'root': './data/Amazon-Photo', 'name': 'Photo'}, 'split_transform': True},
        {'name': 'Amazon-Computers', 'class': Amazon, 'kwargs': {'root': './data/Amazon-Computers', 'name': 'Computers'}, 'split_transform': True},
        {'name': 'Coauthor-CS', 'class': Coauthor, 'kwargs': {'root': './data/Coauthor-CS', 'name': 'CS'}, 'split_transform': True},
        {'name': 'Cora', 'class': Planetoid, 'kwargs': {'root': './data/Cora', 'name': 'Cora'}, 'split_transform': False},
        {'name': 'Citeseer', 'class': Planetoid, 'kwargs': {'root': './data/Citeseer', 'name': 'Citeseer'}, 'split_transform': False},
        {'name': 'PubMed', 'class': Planetoid, 'kwargs': {'root': './data/PubMed', 'name': 'PubMed'}, 'split_transform': False},
    ]

    all_final_results = {}

    for config in dataset_configs:
        dataset_name = config['name']
        dataset_class = config['class']
        kwargs = config['kwargs']
        split_transform = config['split_transform']

        print(f"\n=== Processing {dataset_name} dataset ===")

        transforms_list = [T.NormalizeFeatures()]
        if split_transform:
             print(f"Applying RandomNodeSplit to {dataset_name}")
             transforms_list.append(T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.2))

        transform = T.Compose(transforms_list)

        try:
            dataset = dataset_class(transform=transform, **kwargs)
            data = dataset[0]
            if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
                 print(f"Warning: Dataset {dataset_name} does not have standard train/val/test masks after transforms.")
            if not hasattr(data, 'y') or data.y is None:
                 print(f"Warning: Dataset {dataset_name} does not have labels (data.y). Node classification task cannot proceed.")
                 continue

            data = data.to(device)
            print(f"{dataset_name} loaded successfully. Num nodes: {data.num_nodes}, Features: {dataset.num_features}, Classes: {dataset.num_classes}")
            print(f"Data masks: Train={data.train_mask.sum().item() if hasattr(data, 'train_mask') else 'N/A'}, Val={data.val_mask.sum().item() if hasattr(data, 'val_mask') else 'N/A'}, Test={data.test_mask.sum().item() if hasattr(data, 'test_mask') else 'N/A'}")

        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            print("Skipping this dataset.")
            continue

        if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y'):
             print(f"Error: Loaded data for {dataset_name} is missing required attributes (x, edge_index, or y). Skipping.")
             continue
        if not hasattr(data, 'val_mask') or data.val_mask is None or not data.val_mask.any():
             print(f"Warning: No validation mask available for {dataset_name}. Early stopping will be disabled.")


        in_channels = dataset.num_features
        num_classes = dataset.num_classes

        print(f"\n=== Training final model for {dataset_name} with fixed parameters ===")
        print(f"Params: beta={beta}, mask={mask_min}-{mask_max}, temp={temp}, LRs=G:{lr_graph},B:{lr_bert},O:{lr_other}, Patience={patience}")

        final_model = GraphMaskedLM(in_channels, hidden_channels=hidden_channels, num_classes=num_classes, nhead=nhead).to(device)

        print("Starting final pretraining...")
        pretrain_contrastive(final_model, data, pretrain_epochs=pretrain_epochs, temp=temp,
                            mask_min=mask_min, mask_max=mask_max, beta=beta, verbose=True,
                            lr_graph=lr_graph, lr_bert=lr_bert, lr_other=lr_other, weight_decay=weight_decay)

        print("Starting final fine-tuning...")
        train_model(final_model, data, num_epochs=train_epochs,
                   mask_min=mask_min, mask_max=mask_max, beta=beta, verbose=True,
                   lr_graph=lr_graph, lr_bert=lr_bert, lr_other=lr_other, weight_decay=weight_decay,
                   patience=patience)

        print("Evaluating final model...")
        final_eval_metrics = evaluate_model(final_model, data)
        print(f"Final Evaluation Metrics for {dataset_name}: {final_eval_metrics}")

        all_final_results[dataset_name] = final_eval_metrics

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f'checkpoints/{dataset_name.lower()}_final_model_{timestamp}.pt'
        try:
            torch.save(final_model.state_dict(), final_model_path)
            print(f"Final model saved to {final_model_path}")
        except Exception as e:
            print(f"Error saving final model {final_model_path}: {e}")

    results_filename = f'results/all_datasets_final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    try:
        with open(results_filename, 'w') as f:
            json.dump(all_final_results, f, indent=2)
        print(f"\nFinal evaluation results for all datasets saved to {results_filename}")
    except Exception as e:
        print(f"Error saving final results summary: {e}")

    print("\n=== All dataset processing completed! ===")

if __name__ == "__main__":
    main()