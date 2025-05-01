import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from transformers import AutoModel, AutoTokenizer
import random
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import time
import copy
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DegreeFeatures(object):
    def __init__(self, max_degree=None):
        self.max_degree = max_degree

    def __call__(self, data):
        if data.x is None:
            if data.num_nodes == 0:
                if self.max_degree is None: self.max_degree = 0
                data.x = torch.empty((0, self.max_degree + 1), dtype=torch.float)
                return data
            deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            if self.max_degree is None:
                self.max_degree = int(deg.max().item()) if data.num_nodes > 0 else 0
            deg = torch.clamp(deg, max=self.max_degree)
            deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)
            data.x = deg
        elif data.x.dtype == torch.long:
            data.x = data.x.float()
        return data

def get_max_degree(dataset):
    max_deg = 0
    for data in dataset:
        if data is not None and data.num_nodes > 0 and hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
            row, col = data.edge_index
            if row.numel() > 0:
                 deg = degree(row, data.num_nodes)
                 current_max_deg = int(deg.max().item())
                 max_deg = max(max_deg, current_max_deg)
    return max_deg if max_deg > 0 else 1


def generate_semantic_mask(data, mask_ratio):
    num_nodes = data.num_nodes
    num_mask = max(1, int(mask_ratio * num_nodes))

    if num_nodes == 0:
        dev = data.x.device if hasattr(data, 'x') and data.x is not None else device
        return torch.zeros(num_nodes, dtype=torch.bool, device=dev)

    has_edges = hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0
    dev = data.x.device if hasattr(data, 'x') and data.x is not None else device

    if not has_edges:
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
        num_mask = min(num_mask, num_nodes)
        if num_mask > 0:
            sampled_indices = torch.randperm(num_nodes, device=dev)[:num_mask]
            if sampled_indices.numel() > 0:
                mask[sampled_indices] = True
        return mask

    row, col = data.edge_index
    deg = degree(row, num_nodes=num_nodes) + 1
    deg = deg.to(dev)

    deg_sum = deg.sum().float()
    if deg_sum == 0 or torch.isnan(deg_sum):
        probs = torch.ones(num_nodes, device=dev) / num_nodes if num_nodes > 0 else torch.tensor([], device=dev)
    else:
        probs = deg.float() / deg_sum

    probs = torch.nan_to_num(probs, nan=1.0/num_nodes if num_nodes > 0 else 0.0)
    if probs.sum() <= 0 and num_nodes > 0:
        probs = torch.ones(num_nodes, device=dev) / num_nodes

    num_mask = min(num_mask, num_nodes)

    if num_nodes > 0 and num_mask > 0:
        try:
            probs = torch.clamp(probs, min=1e-9)
            prob_sum = probs.sum()
            if not torch.isclose(prob_sum, torch.tensor(1.0, device=probs.device), atol=1e-5):
                 if prob_sum > 0:
                     probs = probs / prob_sum
                 else:
                     probs = torch.ones_like(probs) / num_nodes

            if probs.sum() <= 0 or torch.isnan(probs).any() or torch.isinf(probs).any():
                print(f"Warning: Invalid probabilities detected before multinomial sampling (sum={probs.sum()}). Falling back to uniform.")
                sampled_indices = torch.randperm(num_nodes, device=probs.device)[:num_mask]
            else:
                non_zero_probs = (probs > 0).sum().item()
                current_num_mask = min(num_mask, non_zero_probs)
                if current_num_mask != num_mask:
                    print(f"Warning: Reducing num_mask from {num_mask} to {current_num_mask} due to zero probabilities.")
                    num_mask = current_num_mask

                if num_mask > 0 :
                     sampled_indices = torch.multinomial(probs, num_mask, replacement=False)
                else:
                     sampled_indices = torch.tensor([], dtype=torch.long, device=probs.device)

        except RuntimeError as e:
            print(f"Warning: Multinomial sampling failed ('{e}'). Falling back to uniform sampling. Probs sum: {probs.sum()}, Num nodes: {num_nodes}, Num mask: {num_mask}")
            num_mask = min(num_mask, num_nodes)
            if num_mask > 0:
                sampled_indices = torch.randperm(num_nodes, device=probs.device)[:num_mask]
            else:
                sampled_indices = torch.tensor([], dtype=torch.long, device=probs.device)

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=probs.device)
        if sampled_indices.numel() > 0:
            mask[sampled_indices] = True
    else:
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=dev)

    return mask


def soft_masking(x, mask, mask_token, beta=0.7):
    device = x.device
    mask = mask.to(device)
    mask_token_on_device = mask_token.to(device)

    if mask_token_on_device.shape[0] != x.shape[1]:
         target_dim = x.shape[1]
         current_dim = mask_token_on_device.shape[0]
         if target_dim > current_dim:
             padding = torch.zeros(target_dim - current_dim, device=device)
             mask_token_resized = torch.cat([mask_token_on_device, padding])
         else:
             mask_token_resized = mask_token_on_device[:target_dim]
         mask_token_expanded = mask_token_resized.unsqueeze(0)
    else:
        mask_token_expanded = mask_token_on_device.unsqueeze(0)


    x_masked = x.clone()
    if mask.any():
        if mask.shape[0] == x.shape[0]:
            x_masked[mask] = (1 - beta) * x[mask] + beta * mask_token_expanded
        else:
             print(f"Warning: Mask shape {mask.shape} incompatible with x shape {x.shape}. Skipping masking for this batch.")

    return x_masked

def nt_xent_loss(z1, z2, temperature=0.5, batch_size=None):
    device = z1.device
    total_loss = 0.0
    num_batches = 0
    total_samples = z1.size(0)

    if total_samples <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    effective_batch_size = min(batch_size, total_samples) if batch_size else total_samples

    for i in range(0, total_samples, effective_batch_size):
        end_idx = min(i + effective_batch_size, total_samples)
        batch_size_curr = end_idx - i

        if batch_size_curr <= 1:
            continue

        batch_z1 = F.normalize(z1[i:end_idx], dim=1)
        batch_z2 = F.normalize(z2[i:end_idx], dim=1)

        batch_emb = torch.cat([batch_z1, batch_z2], dim=0)

        similarity = torch.mm(batch_emb.float(), batch_emb.t().float()) / temperature

        mask = torch.eye(2 * batch_size_curr, dtype=torch.bool, device=device)
        similarity_neg_filled = similarity.masked_fill(mask, -float('inf'))

        pos_indices = torch.arange(batch_size_curr, device=device)
        labels = torch.cat([pos_indices + batch_size_curr, pos_indices], dim=0)

        batch_loss = F.cross_entropy(similarity_neg_filled, labels, reduction='mean')

        total_loss += batch_loss
        num_batches += 1

    if num_batches == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / num_batches


class GraphMaskedLM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, nhead=4, model_name='distilbert/distilbert-base-uncased', pool_type='mean'):
        super(GraphMaskedLM, self).__init__()
        self.in_channels = in_channels
        self.mask_token = nn.Parameter(torch.zeros(in_channels))
        nn.init.xavier_uniform_(self.mask_token.data.unsqueeze(0))

        self.gat1 = GATConv(in_channels, hidden_channels, heads=nhead, dropout=0.6)
        self.ln1 = nn.LayerNorm(hidden_channels * nhead)
        self.dropout1 = nn.Dropout(0.3)
        self.gat2 = GATConv(hidden_channels * nhead, hidden_channels, heads=2, concat=True, dropout=0.6)
        self.ln2 = nn.LayerNorm(hidden_channels * 2)
        self.dropout2 = nn.Dropout(0.3)

        print(f"Loading HuggingFace model: {model_name}")
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_encoder = AutoModel.from_pretrained(model_name)
            bert_hidden_size = self.bert_encoder.config.hidden_size
        except Exception as e:
            print(f"Error loading HuggingFace model {model_name}: {e}. Please check model name and internet connection.")
            raise e

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

        self.pool_type = pool_type
        if pool_type == 'mean': self.pool = global_mean_pool
        elif pool_type == 'add': self.pool = global_add_pool
        elif pool_type == 'max': self.pool = global_max_pool
        else: raise ValueError(f"Unsupported pool type: {pool_type}")

        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, num_classes)
        )

    def ensure_mask_token_dim(self, feature_dim):
        if self.mask_token.shape[0] != feature_dim:
            print(f"Re-initializing mask token from dim {self.mask_token.shape[0]} to {feature_dim}")
            current_device = self.mask_token.device
            self.mask_token = nn.Parameter(torch.zeros(feature_dim, device=current_device))
            nn.init.xavier_uniform_(self.mask_token.data.unsqueeze(0))
            if self.gat1.in_channels != feature_dim:
                 print(f"Updating GAT1 input channels from {self.gat1.in_channels} to {feature_dim}")
                 hidden_channels = self.gat1.out_channels // self.gat1.heads
                 nhead = self.gat1.heads
                 dropout = self.gat1.dropout
                 self.gat1 = GATConv(feature_dim, hidden_channels, heads=nhead, dropout=dropout).to(current_device)

            self.in_channels = feature_dim


    def get_graph_node_embeddings(self, x, edge_index):
        num_nodes = x.size(0)
        gat_out_dim = self.gat2.out_channels

        if num_nodes == 0:
            return torch.zeros(0, self.proj.out_features, device=x.device)

        has_edges = edge_index is not None and edge_index.numel() > 0
        if not has_edges:
            pass

        try:
            x = self.gat1(x, edge_index)
            x = self.ln1(x)
            x = F.elu(x)
            x = self.dropout1(x)
            x = self.gat2(x, edge_index)
            x = self.ln2(x)
            x = F.elu(x)
            x = self.dropout2(x)

            if torch.isnan(x).any() or torch.isinf(x).any():
                 print("Warning: NaNs/Infs detected after GAT layers. Returning zero embeddings projected.")
                 return torch.zeros(num_nodes, self.proj.out_features, device=x.device)

            x = self.proj(x)
            return x

        except Exception as e:
             print(f"Error during GAT processing: {e}. Returning zero embeddings projected.")
             return torch.zeros(num_nodes, self.proj.out_features, device=x.device)


    def forward(self, x, edge_index, batch, node_mask):
        current_device = x.device
        self.ensure_mask_token_dim(x.shape[1])

        x_graph = self.get_graph_node_embeddings(x, edge_index)

        texts = ["[MASK]" if flag else "node" for flag in node_mask.cpu().tolist()]
        bert_hidden_size = self.bert_encoder.config.hidden_size

        bert_embeddings = torch.zeros(0, bert_hidden_size, device=current_device)
        if texts:
            self.bert_encoder.to(current_device)
            with torch.set_grad_enabled(self.training):
                try:
                    encodings = self.bert_tokenizer.batch_encode_plus(
                        texts, padding='longest', truncation=True, max_length=16, return_tensors="pt"
                    ).to(current_device)
                    bert_output = self.bert_encoder(**encodings)
                    bert_embeddings = bert_output.last_hidden_state[:, 0, :]
                except Exception as e:
                    print(f"Error during BERT encoding/forward pass: {e}. Using zero BERT embeddings.")
                    bert_embeddings = torch.zeros(len(texts), bert_hidden_size, device=current_device)


        num_graph_nodes = x_graph.size(0)
        num_bert_nodes = bert_embeddings.size(0)

        if num_graph_nodes != num_bert_nodes:
            print(f"Warning: Mismatch GAT nodes ({num_graph_nodes}) vs BERT nodes ({num_bert_nodes}). Attempting alignment.")
            expected_nodes = x.size(0)
            if x_graph.size(0) != expected_nodes:
                print(f"  GAT output size {x_graph.size(0)} differs from input {expected_nodes}. Using zeros.")
                x_graph = torch.zeros(expected_nodes, bert_hidden_size, device=current_device)
            if bert_embeddings.size(0) != expected_nodes:
                 print(f"  BERT output size {bert_embeddings.size(0)} differs from input {expected_nodes}. Using zeros.")
                 bert_embeddings = torch.zeros(expected_nodes, bert_hidden_size, device=current_device)
            num_graph_nodes = expected_nodes


        if x_graph.size(0) != bert_embeddings.size(0):
             print(f"ERROR: Unrecoverable mismatch after alignment GAT ({x_graph.size(0)}) vs BERT ({bert_embeddings.size(0)}). Cannot proceed.")
             num_graphs_in_batch = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 0
             return torch.zeros(num_graphs_in_batch, self.classifier[-1].out_features, device=current_device), \
                    torch.zeros(0, bert_hidden_size, device=current_device)

        if num_graph_nodes > 0 :
            combined = torch.cat([x_graph, bert_embeddings], dim=1)
            fused_node_embeddings = self.fusion_network(combined)

            if batch is None or batch.numel() != fused_node_embeddings.size(0):
                 print(f"Warning: Batch tensor invalid (size {batch.numel() if batch is not None else 'None'}) for pooling {fused_node_embeddings.size(0)} nodes. Using mean pooling over all nodes.")
                 graph_embedding = fused_node_embeddings.mean(dim=0, keepdim=True)
            else:
                 graph_embedding = self.pool(fused_node_embeddings, batch)

        else:
            fused_node_embeddings = torch.zeros(0, bert_hidden_size, device=current_device)
            num_graphs_in_batch = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 0
            graph_embedding = torch.zeros(num_graphs_in_batch, bert_hidden_size, device=current_device)


        if graph_embedding.size(0) > 0:
             logits = self.classifier(graph_embedding)
        else:
             logits = torch.zeros(0, self.classifier[-1].out_features, device=current_device)


        return logits, fused_node_embeddings


def pretrain_contrastive(model, loader, pretrain_epochs=20, temp=0.5, mask_min=0.2, mask_max=0.4, beta=0.7, verbose=True, contrastive_batch_size=None):
    model.train()
    initial_feature_dim = None
    try:
        temp_iter = iter(loader)
        while True:
             first_batch = next(temp_iter)
             if first_batch is not None and hasattr(first_batch, 'x') and first_batch.x is not None and first_batch.x.numel() > 0:
                 initial_feature_dim = first_batch.x.shape[1]
                 break
        del temp_iter
        if initial_feature_dim is not None:
             model.ensure_mask_token_dim(initial_feature_dim)
             print(f"Pretrain: Ensured mask token dim is {initial_feature_dim}")
        else:
            print("Warning: Could not determine feature dimension from loader for pre-emptive mask token resize (loader might be empty or contain only invalid graphs).")

    except StopIteration:
        print("Warning: Pretrain loader is empty. Cannot verify mask token dimension or run pretraining.")
        return 0.0
    except Exception as e:
        print(f"Warning: Error accessing pretrain loader: {e}. Cannot verify mask token dimension.")


    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs, eta_min=1e-6)

    total_loss_epoch = []
    start_time = time.time()

    print(f"Starting contrastive pretraining for {pretrain_epochs} epochs...")
    for epoch in range(pretrain_epochs):
        epoch_loss = 0.0
        num_batches = 0
        batch_iter = iter(loader)

        while True:
            try:
                batch_data = next(batch_iter)
                batch_data = batch_data.to(device)

                if not (hasattr(batch_data, 'x') and batch_data.x is not None and batch_data.x.numel() > 0 and batch_data.num_nodes > 0):
                    continue

                model.ensure_mask_token_dim(batch_data.x.shape[1])

                optimizer.zero_grad()

                mask_ratio1 = random.uniform(mask_min, mask_max)
                mask_ratio2 = random.uniform(mask_min, mask_max)
                node_mask1 = generate_semantic_mask(batch_data, mask_ratio1)
                node_mask2 = generate_semantic_mask(batch_data, mask_ratio2)

                x1 = soft_masking(batch_data.x, node_mask1, model.mask_token, beta=beta)
                x2 = soft_masking(batch_data.x, node_mask2, model.mask_token, beta=beta)

                g1 = model.get_graph_node_embeddings(x1, batch_data.edge_index)
                g2 = model.get_graph_node_embeddings(x2, batch_data.edge_index)

                if g1.numel() == 0 or g2.numel() == 0 or g1.shape[0] != g2.shape[0]:
                    continue

                loss = nt_xent_loss(g1, g2, temperature=temp, batch_size=contrastive_batch_size)

                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected in pretraining epoch {epoch}, batch {num_batches}. Skipping update.")
                    continue

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    pass


                epoch_loss += loss.item()
                num_batches += 1

            except StopIteration:
                break
            except Exception as e:
                print(f"Error during pretrain batch processing (Epoch {epoch}, Batch {num_batches}): {e}. Skipping batch.")
                traceback.print_exc()
                continue

        scheduler.step()
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        total_loss_epoch.append(avg_epoch_loss)

        if verbose and (epoch + 1) % 5 == 0:
            elapsed_time = time.time() - start_time
            print(f"[Pretrain] Epoch: {epoch+1}/{pretrain_epochs}, Avg Loss: {avg_epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Time: {elapsed_time:.2f}s")

        if np.isnan(avg_epoch_loss) and num_batches > 0:
            print(f"ERROR: Average pretrain loss became NaN in epoch {epoch+1}. Stopping pretraining early.")
            break

    final_avg_loss = np.nanmean(total_loss_epoch[-5:]) if total_loss_epoch else 0.0
    print(f"[Pretrain] Final Avg Loss (last 5 epochs): {final_avg_loss:.4f}")
    print(f"Pretraining finished in {time.time() - start_time:.2f} seconds.")
    return final_avg_loss


def train_model(model, train_loader, val_loader, num_epochs, mask_min, mask_max, beta, patience, weight_decay=0.01, verbose=True):
    criterion = torch.nn.CrossEntropyLoss()
    initial_feature_dim = None

    try:
        temp_iter = iter(train_loader)
        while True:
            first_batch = next(temp_iter)
            if first_batch is not None and hasattr(first_batch, 'x') and first_batch.x is not None and first_batch.x.numel() > 0:
                 initial_feature_dim = first_batch.x.shape[1]
                 break
        del temp_iter
        if initial_feature_dim is not None:
             model.ensure_mask_token_dim(initial_feature_dim)
             print(f"Train: Ensured mask token dim is {initial_feature_dim}")
        else:
            print("Warning: Could not determine feature dimension from train_loader for pre-emptive mask token resize.")

    except StopIteration:
        print("Error: Train loader is empty. Cannot train the model.")
        return model, -1
    except Exception as e:
        print(f"Warning: Error accessing train loader: {e}. Cannot verify mask token dimension.")


    lr_graph = 1e-3
    lr_bert = 1e-5
    lr_other = 1e-4

    graph_params = []
    bert_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('bert_encoder.'):
            bert_params.append(param)
        elif name.startswith('gat1.') or name.startswith('ln1.') or \
            name.startswith('gat2.') or name.startswith('ln2.') or \
            name.startswith('proj.'):
            graph_params.append(param)
        else:
            other_params.append(param)

    optimizer_grouped_parameters = [
        {'params': graph_params, 'lr': lr_graph, 'weight_decay': weight_decay},
        {'params': bert_params, 'lr': lr_bert, 'weight_decay': weight_decay},
        {'params': other_params, 'lr': lr_other, 'weight_decay': weight_decay}
    ]

    print(f"Optimizer groups: Graph ({len(graph_params)} tensors, lr={lr_graph}), BERT ({len(bert_params)} tensors, lr={lr_bert}), Other ({len(other_params)} tensors, lr={lr_other})")

    optimizer = optim.AdamW(optimizer_grouped_parameters)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_metric = -1.0
    best_model_state_dict = None
    epochs_no_improve = 0
    best_epoch = -1
    eval_freq = 1

    start_time = time.time()
    print(f"Starting training for up to {num_epochs} epochs with patience {patience}.")
    use_validation = val_loader is not None
    if use_validation:
        print("Using Validation set accuracy for early stopping and best model selection.")
    else:
        print("No Validation set provided. Using Training accuracy for early stopping and best model selection.")


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        epoch_train_preds = []
        epoch_train_labels = []

        batch_iter = iter(train_loader)
        while True:
            try:
                batch_data = next(batch_iter)
                batch_data = batch_data.to(device)

                if not (hasattr(batch_data, 'x') and batch_data.x is not None and batch_data.x.numel() > 0 and
                        hasattr(batch_data, 'y') and batch_data.y is not None and batch_data.num_nodes > 0):
                    continue

                model.ensure_mask_token_dim(batch_data.x.shape[1])

                optimizer.zero_grad()

                mask_ratio = random.uniform(mask_min, mask_max)
                node_mask = generate_semantic_mask(batch_data, mask_ratio)
                x_soft = soft_masking(batch_data.x, node_mask, model.mask_token, beta=beta)

                graph_logits, _ = model(x_soft, batch_data.edge_index, batch_data.batch, node_mask)

                if graph_logits.size(0) != batch_data.y.size(0) or graph_logits.size(0) == 0:
                    print(f"Warning: Logits size ({graph_logits.size()}) mismatch labels ({batch_data.y.size()}) or zero size in train epoch {epoch}. Skipping batch.")
                    continue
                if torch.isnan(graph_logits).any() or torch.isinf(graph_logits).any():
                     print(f"Warning: NaN/Inf detected in logits in train epoch {epoch}. Skipping batch.")
                     continue


                loss = criterion(graph_logits, batch_data.y)

                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected in training epoch {epoch}, batch {num_batches}. Skipping update.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

                pred = graph_logits.max(dim=1)[1]
                epoch_train_preds.append(pred.cpu())
                epoch_train_labels.append(batch_data.y.cpu())

                num_batches += 1

            except StopIteration:
                break
            except Exception as e:
                print(f"Error during training batch processing (Epoch {epoch}, Batch {num_batches}): {e}. Skipping batch.")
                traceback.print_exc()
                continue

        scheduler.step()
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time

        current_metric_value = -1.0
        metric_name = "N/A"
        f1_val = 0.0

        perform_check = (epoch + 1) % eval_freq == 0 or epoch == num_epochs - 1

        if perform_check:
            train_acc = 0.0
            if epoch_train_labels and epoch_train_preds:
                try:
                    epoch_train_labels_np = torch.cat(epoch_train_labels).numpy()
                    epoch_train_preds_np = torch.cat(epoch_train_preds).numpy()
                    if len(epoch_train_labels_np) > 0:
                        train_acc = accuracy_score(epoch_train_labels_np, epoch_train_preds_np) * 100
                except Exception as e:
                    print(f"Warning: Error calculating training accuracy for epoch {epoch+1}: {e}")


            if use_validation:
                val_results = evaluate_model(model, val_loader, set_name="Validation")
                if val_results:
                    current_metric_value = val_results['accuracy']
                    f1_val = val_results['f1']
                    metric_name = "Val Acc"
                    current_lrs = [group['lr'] for group in optimizer.param_groups]
                    lr_str = f"LRs: G={current_lrs[0]:.1E}, B={current_lrs[1]:.1E}, O={current_lrs[2]:.1E}"
                    if verbose:
                         print(f"  [Epoch {epoch+1}/{num_epochs}] TrainLoss: {avg_epoch_loss:.4f}, TrainAcc: {train_acc:.2f}%, ValAcc: {current_metric_value:.2f}%, ValF1: {f1_val:.4f}, {lr_str}, Time: {epoch_duration:.2f}s")
                else:
                    print(f"  [Epoch {epoch+1}/{num_epochs}] TrainLoss: {avg_epoch_loss:.4f}, TrainAcc: {train_acc:.2f}%, Validation Failed! Time: {epoch_duration:.2f}s")
                    current_metric_value = train_acc
                    metric_name = "Train Acc (Val Failed)"
            else:
                current_metric_value = train_acc
                metric_name = "Train Acc"
                current_lrs = [group['lr'] for group in optimizer.param_groups]
                lr_str = f"LRs: G={current_lrs[0]:.1E}, B={current_lrs[1]:.1E}, O={current_lrs[2]:.1E}"
                if verbose:
                     print(f"  [Train Epoch {epoch+1}/{num_epochs}] AvgLoss: {avg_epoch_loss:.4f}, Train Acc: {current_metric_value:.2f}%, {lr_str}, Time: {epoch_duration:.2f}s")

            if current_metric_value > best_metric:
                best_metric = current_metric_value
                best_epoch = epoch + 1
                epochs_no_improve = 0
                try:
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    if verbose: print(f"    [*] New best {metric_name}: {best_metric:.2f}% at epoch {best_epoch}.")
                except Exception as e:
                    print(f"Error during model state deepcopy: {e}. Best model state not saved.")
                    best_model_state_dict = None
            else:
                epochs_no_improve += eval_freq
                if verbose: print(f"    [!] No improvement in {metric_name} for {epochs_no_improve} cumulative checks.")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in {metric_name} for {patience} checks.")
                break

        if np.isnan(avg_epoch_loss) and num_batches > 0:
            print(f"ERROR: Average train loss became NaN in epoch {epoch+1}. Stopping training early.")
            break

    training_duration = time.time() - start_time
    print(f"Training finished in {training_duration:.2f} seconds.")

    if best_model_state_dict is not None:
        metric_source = "Validation" if use_validation else "Training"
        print(f"Loading best model state from epoch {best_epoch} with best {metric_source} Acc: {best_metric:.2f}%")
        try:
            model.load_state_dict(best_model_state_dict)
        except Exception as e:
            print(f"Error loading best model state dict: {e}. Using final model state instead.")
    else:
        print(f"Warning: No best model state was saved (or patience=0/error). Using final model state from epoch {epoch+1}.")

    return model, best_epoch


def evaluate_model(model, loader, set_name="Test"):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    start_time = time.time()

    with torch.no_grad():
        batch_iter = iter(loader)
        while True:
            try:
                batch_data = next(batch_iter)
                batch_data = batch_data.to(device)

                if not (hasattr(batch_data, 'x') and batch_data.x is not None and batch_data.x.numel() > 0 and
                        hasattr(batch_data, 'y') and batch_data.y is not None and batch_data.num_nodes > 0):
                    continue

                eval_mask = torch.zeros(batch_data.num_nodes, dtype=torch.bool, device=device)

                model.ensure_mask_token_dim(batch_data.x.shape[1])

                graph_logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch, eval_mask)

                if graph_logits.size(0) != batch_data.y.size(0) or graph_logits.size(0) == 0:
                    print(f"Warning: Logits size ({graph_logits.size()}) mismatch labels ({batch_data.y.size()}) or zero size in {set_name} eval. Skipping batch.")
                    continue
                if torch.isnan(graph_logits).any() or torch.isinf(graph_logits).any():
                     print(f"Warning: NaN/Inf detected in logits during {set_name} eval. Skipping batch.")
                     continue

                if batch_data.y is not None:
                    try:
                        loss = criterion(graph_logits, batch_data.y)
                        total_loss += loss.item()
                    except Exception as e:
                        print(f"Warning: Error calculating loss in {set_name} eval for batch {num_batches}: {e}.")

                pred = graph_logits.max(dim=1)[1]
                all_preds.append(pred.cpu())
                all_labels.append(batch_data.y.cpu())
                num_batches += 1

            except StopIteration:
                break
            except Exception as e:
                print(f"Error during {set_name} evaluation batch processing (Batch {num_batches}): {e}. Skipping batch.")
                traceback.print_exc()
                continue

    if num_batches == 0:
        print(f"Warning: No batches were successfully evaluated for the {set_name} set.")
        return None

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if not all_labels or not all_preds:
         print(f"Warning: No labels or predictions collected during {set_name} evaluation after processing batches.")
         return {'loss': avg_loss, 'accuracy': 0.0, 'f1': 0.0}


    try:
        all_preds_tensor = torch.cat(all_preds)
        all_labels_tensor = torch.cat(all_labels)

        if all_labels_tensor.numel() == 0 or all_preds_tensor.numel() == 0:
            print(f"Warning: Labels or predictions are empty after concatenation during {set_name} evaluation.")
            return {'loss': avg_loss, 'accuracy': 0.0, 'f1': 0.0}

        all_preds_np = all_preds_tensor.numpy()
        all_labels_np = all_labels_tensor.numpy()

        acc = accuracy_score(all_labels_np, all_preds_np) * 100
        f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)

    except Exception as e:
        print(f"Error calculating metrics (Acc/F1) for {set_name}: {e}")
        acc = 0.0
        f1 = 0.0

    eval_duration = time.time() - start_time

    return {'loss': avg_loss, 'accuracy': acc, 'f1': f1}


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    num_final_runs = 5
    beta = 0.7
    mask_min = 0.2
    mask_max = 0.4
    pretrain_epochs = 30
    train_epochs = 120
    patience = 30
    batch_size = 32
    contrastive_batch_size = 32
    temp = 0.8
    hidden_channels = 128
    nhead = 4
    pool_type = 'mean'
    bert_model_name = 'distilbert/distilbert-base-uncased'
    split_seed = 42
    weight_decay = 0.01


    dataset_configs = [
        #{'name': 'PROTEINS', 'root': './data/TUDataset'},
        #{'name': 'MUTAG', 'root': './data/TUDataset'},
        {'name': 'REDDIT-BINARY', 'root': './data/TUDataset'}
    ]

    all_datasets_results = {}
    dataset_splits_info = {}

    for config in dataset_configs:
        dataset_name = config['name']
        dataset_root = config['root']

        print(f"\n{'='*25} Processing Dataset: {dataset_name} {'='*25}")

        try:
            print("Loading raw dataset info to determine features and classes...")
            raw_dataset_info = TUDataset(root=dataset_root, name=dataset_name, use_node_attr=True)
            num_classes = raw_dataset_info.num_classes

            valid_graphs_info = [g for g in raw_dataset_info if g is not None and hasattr(g, 'num_nodes') and hasattr(g, 'edge_index')]
            if not valid_graphs_info:
                print(f"Error: No valid graphs found in raw dataset {dataset_name} after basic filtering. Skipping.")
                all_datasets_results[dataset_name] = None
                continue

            has_node_attributes = False
            example_feature_dim = -1
            for g in valid_graphs_info:
                if hasattr(g, 'x') and g.x is not None and g.x.numel() > 0:
                    has_node_attributes = True
                    example_feature_dim = g.x.shape[1]
                    break

            max_degree = get_max_degree(valid_graphs_info)
            del raw_dataset_info, valid_graphs_info

            print(f"Max degree computed: {max_degree}")
            if has_node_attributes:
                print(f"Dataset has original node features (example dim: {example_feature_dim}). Will use them.")
                feature_source = "Original Features"
            else:
                print(f"Dataset has no original node features. Adding degree features (dim: {max_degree + 1}).")
                feature_source = "Degree Features"

            transform_list = []
            if not has_node_attributes:
                transform_list.append(DegreeFeatures(max_degree=max_degree))
            pre_transform = T.Compose(transform_list) if transform_list else None


            print("Loading dataset with appropriate pre-transform...")
            dataset = TUDataset(root=dataset_root, name=dataset_name,
                                pre_transform=pre_transform,
                                use_node_attr=True,
                                )

            full_dataset = []
            in_channels = -1
            for i, data in enumerate(dataset):
                valid = True
                if data is None:
                    valid = False
                elif not hasattr(data, 'x') or data.x is None or data.x.numel() == 0:
                    valid = False
                elif not hasattr(data, 'y') or data.y is None:
                    valid = False
                elif not hasattr(data, 'edge_index'):
                    valid = False

                if valid:
                    current_feat_dim = data.num_features
                    if in_channels == -1:
                        in_channels = current_feat_dim
                    elif in_channels != current_feat_dim:
                        print(f"Warning: Inconsistent feature dimension found in {dataset_name}. Expected {in_channels}, got {current_feat_dim} in graph {i}. Skipping graph.")
                        valid = False

                if valid and in_channels <= 0:
                    print(f"Warning: Graph {i} has 0 feature dimension after processing. Skipping graph.")
                    valid = False

                if valid:
                    full_dataset.append(data)


            print(f"Number of graphs after loading and filtering: {len(full_dataset)}")

            if not full_dataset:
                print(f"Error: Dataset {dataset_name} is empty after filtering. Check data integrity or pre-processing steps.")
                all_datasets_results[dataset_name] = None
                continue

            if in_channels <= 0:
                print(f"Error: Determined input channel size is {in_channels}. Cannot proceed. Check feature generation/loading.")
                all_datasets_results[dataset_name] = None
                continue

            print(f"Dataset Stats: Graphs={len(full_dataset)}, Features={in_channels} ({feature_source}), Classes={num_classes}")

        except Exception as e:
            print(f"Error preparing dataset {dataset_name}: {e}. Skipping this dataset.")
            traceback.print_exc()
            all_datasets_results[dataset_name] = None
            continue

        train_dataset, val_dataset, test_dataset = [], [], []
        val_loader = None

        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))
        np.random.seed(split_seed)
        np.random.shuffle(indices)

        if dataset_name == 'PROTEINS':
            target_train_size = 889
            target_val_size = 112
            target_test_size = 112
            target_total = target_train_size + target_val_size + target_test_size

            if dataset_size != target_total:
                print(f"Warning: Actual dataset size ({dataset_size}) differs from target ({target_total}) for PROTEINS split. Using absolute counts.")
                target_test_size = min(target_test_size, dataset_size)
                target_val_size = min(target_val_size, dataset_size - target_test_size)
                target_train_size = min(target_train_size, dataset_size - target_test_size - target_val_size)
                print(f"Adjusted counts: Train={target_train_size}, Val={target_val_size}, Test={target_test_size}")


            if target_train_size + target_val_size + target_test_size > dataset_size:
                print(f"Error: Requested split sizes ({target_train_size}+{target_val_size}+{target_test_size}) exceed dataset size ({dataset_size}). Skipping {dataset_name}.")
                all_datasets_results[dataset_name] = None
                continue

            test_indices = indices[:target_test_size]
            val_indices = indices[target_test_size : target_test_size + target_val_size]
            train_indices = indices[target_test_size + target_val_size : target_test_size + target_val_size + target_train_size]

            train_dataset = [full_dataset[i] for i in train_indices]
            val_dataset = [full_dataset[i] for i in val_indices]
            test_dataset = [full_dataset[i] for i in test_indices]

            split_desc = f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} Split"
            print(f"Created {split_desc} for {dataset_name} (absolute counts).")
            print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

        elif dataset_name == 'ENZYMES':
            print(f"Creating 80/10/10 train/validation/test split for {dataset_name}...")
            try:
                train_indices, temp_val_test_indices = train_test_split(
                    indices, test_size=0.2, shuffle=False, random_state=split_seed
                )
                val_indices, test_indices = train_test_split(
                    temp_val_test_indices, test_size=0.5, shuffle=False, random_state=split_seed + 1
                )

                train_dataset = [full_dataset[i] for i in train_indices]
                val_dataset = [full_dataset[i] for i in val_indices]
                test_dataset = [full_dataset[i] for i in test_indices]

                split_desc = "80%/10%/10% Split"
                print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
            except ValueError as e:
                print(f"Error during 80/10/10 split for {dataset_name}: {e}. Dataset might be too small. Skipping.")
                all_datasets_results[dataset_name] = None
                continue

        else:
            current_test_split_ratio = 0.1
            split_desc = f"{int((1-current_test_split_ratio)*100)}%/{int(current_test_split_ratio*100)}% Split (No Validation Set)"
            print(f"Creating {split_desc} for {dataset_name}...")
            try:
                train_indices, test_indices = train_test_split(
                    indices, test_size=current_test_split_ratio, shuffle=False, random_state=split_seed
                )
                train_dataset = [full_dataset[i] for i in train_indices]
                test_dataset = [full_dataset[i] for i in test_indices]
                val_dataset = None

                print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
            except ValueError as e:
                print(f"Error during {split_desc} split for {dataset_name}: {e}. Dataset might be too small. Skipping.")
                all_datasets_results[dataset_name] = None
                continue

        dataset_splits_info[dataset_name] = split_desc

        if not train_dataset:
            print(f"Error: Training set is empty after split for {dataset_name}. Skipping.")
            all_datasets_results[dataset_name] = None
            continue
        if not test_dataset:
            print(f"Error: Test set is empty after split for {dataset_name}. Skipping.")
            all_datasets_results[dataset_name] = None
            continue
        if dataset_name in ['ENZYMES', 'PROTEINS'] and not val_dataset:
            print(f"Error: {dataset_name} split resulted in an empty validation set where one was expected. Skipping.")
            all_datasets_results[dataset_name] = None
            continue


        pre_trained_state_dict = None
        if pretrain_epochs > 0:
            print(f"\nPre-training on the training set ({len(train_dataset)} graphs)...")
            pretrain_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            pretrain_model = GraphMaskedLM(in_channels=in_channels, hidden_channels=hidden_channels,
                                        num_classes=num_classes, nhead=nhead, pool_type=pool_type,
                                        model_name=bert_model_name).to(device)

            pretrain_contrastive(pretrain_model, pretrain_loader, pretrain_epochs=pretrain_epochs,
                                temp=temp, mask_min=mask_min, mask_max=mask_max, beta=beta,
                                verbose=True, contrastive_batch_size=contrastive_batch_size)

            pre_trained_state_dict = copy.deepcopy(pretrain_model.state_dict())
            del pretrain_model, pretrain_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print("\nSkipping Pre-training.")


        print(f"\nStarting {num_final_runs} final training runs...")
        test_metrics_runs = {'accuracy': [], 'f1': [], 'loss': []}

        for run in range(num_final_runs):
            print(f"  --- Run {run + 1}/{num_final_runs} ---")
            run_start_time = time.time()

            model = GraphMaskedLM(in_channels=in_channels, hidden_channels=hidden_channels,
                                num_classes=num_classes, nhead=nhead, pool_type=pool_type,
                                model_name=bert_model_name).to(device)

            if pre_trained_state_dict:
                try:
                    model.load_state_dict(pre_trained_state_dict)
                    print("    Loaded pre-trained weights successfully.")
                except Exception as e:
                    print(f"    Warning: Failed to load pre-trained weights for run {run+1}: {e}. Check layer dimensions. Training from scratch.")
                    model.ensure_mask_token_dim(in_channels)
            else:
                print("    Training from scratch (no pre-trained weights).")
                model.ensure_mask_token_dim(in_channels)


            train_run_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            val_run_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


            print(f"    Training on {len(train_dataset)} graphs (Max Epochs: {train_epochs}, Patience: {patience})...")
            model, best_ep = train_model(
                model, train_run_loader, val_run_loader,
                num_epochs=train_epochs, mask_min=mask_min, mask_max=mask_max,
                beta=beta, patience=patience, weight_decay=weight_decay,
                verbose=True # Set verbose=True for detailed training logs
            )
            metric_source = "validation" if val_run_loader else "training"
            print(f"    Training complete. Best model from epoch {best_ep} (based on {metric_source} accuracy).")

            print(f"    Evaluating run {run+1} on test set ({len(test_dataset)} graphs)...")
            test_metrics = evaluate_model(model, test_loader, set_name=f"Test (Run {run+1})")

            if test_metrics:
                print(f"    [Run {run+1} Test Results] Acc: {test_metrics['accuracy']:.2f}%, F1: {test_metrics['f1']:.4f}, Loss: {test_metrics['loss']:.4f}")
                test_metrics_runs['accuracy'].append(test_metrics['accuracy'])
                test_metrics_runs['f1'].append(test_metrics['f1'])
                test_metrics_runs['loss'].append(test_metrics['loss'])
            else:
                print(f"    Error: Test evaluation failed for run {run+1}. Run results not recorded.")

            run_duration = time.time() - run_start_time
            print(f"  --- Run {run + 1} finished in {run_duration:.2f} seconds ---")

            del model, train_run_loader, val_run_loader, test_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


        if test_metrics_runs['accuracy']:
            num_successful_runs = len(test_metrics_runs['accuracy'])
            print(f"\n--- Results Summary for {dataset_name} ({num_successful_runs}/{num_final_runs} successful runs) ---")
            try:
                acc_mean = np.mean(test_metrics_runs['accuracy'])
                acc_std = np.std(test_metrics_runs['accuracy'])
                f1_mean = np.mean(test_metrics_runs['f1'])
                f1_std = np.std(test_metrics_runs['f1'])
                loss_mean = np.mean(test_metrics_runs['loss'])
                loss_std = np.std(test_metrics_runs['loss'])

                dataset_summary = {
                    'acc_mean': acc_mean, 'acc_std': acc_std,
                    'f1_mean': f1_mean, 'f1_std': f1_std,
                    'loss_mean': loss_mean, 'loss_std': loss_std,
                    'num_successful_runs': num_successful_runs
                }
                print(f"  Average Accuracy: {acc_mean:.2f}% +/- {acc_std:.2f}%")
                print(f"  Average F1 Score: {f1_mean:.4f} +/- {f1_std:.4f}")
                all_datasets_results[dataset_name] = dataset_summary
            except Exception as e:
                print(f"Error calculating summary statistics for {dataset_name}: {e}")
                all_datasets_results[dataset_name] = None

        else:
            print(f"\nDataset {dataset_name}: No successful runs completed evaluation.")
            all_datasets_results[dataset_name] = None


    print("\n\n" + "="*45)
    print("========= Final Summary Across All Datasets =========")
    print("="*45)
    for name, summary_stats in all_datasets_results.items():
        split_info = dataset_splits_info.get(name, "Unknown Split Info")
        print(f"\n--- Dataset: {name} ({split_info}) ---")
        if summary_stats:
            num_runs_reported = summary_stats['num_successful_runs']
            print(f"  (Based on {num_runs_reported}/{num_final_runs} successful runs)")
            print(f"  Accuracy: {summary_stats['acc_mean']:.2f}% +/- {summary_stats['acc_std']:.2f}%")
            print(f"  F1 Score: {summary_stats['f1_mean']:.4f} +/- {summary_stats['f1_std']:.4f}")
        else:
            print(f"  Failed to produce results or processing aborted.")
    print("\n" + "="*45)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    main()