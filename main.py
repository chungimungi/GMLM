import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.nn import RGCNConv, GraphNorm
from torch_geometric.utils import degree
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import random
import numpy as np
import itertools
import os
import json
import logging
from datetime import datetime
from sklearn.metrics import f1_score
import copy
from tqdm import tqdm, trange
import time
from torch_geometric.data import Data
from torch.utils.checkpoint import checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def setup_logging(log_level=logging.INFO):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def generate_active_node_mask(data, mask_ratio, split_edge=None, base_mask_name='train_mask'):
    if base_mask_name and hasattr(data, base_mask_name) and getattr(data, base_mask_name) is not None:
        base_nodes_idx = getattr(data, base_mask_name).nonzero(as_tuple=False).reshape(-1)
        if base_nodes_idx.numel() == 0:
            return torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    elif split_edge is not None and 'train' in split_edge and 'edge' in split_edge['train']:
        train_edges = split_edge['train']['edge']
        base_nodes_idx = torch.unique(train_edges.flatten())
    else:
        base_nodes_idx = torch.arange(data.num_nodes, device=data.x.device)

    num_base_nodes = base_nodes_idx.size(0)
    if num_base_nodes == 0:
        return torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)

    num_select = max(1, min(int(mask_ratio * num_base_nodes), num_base_nodes))

    edge_index = data.edge_index.to(torch.long)
    deg = degree(edge_index[0], num_nodes=data.num_nodes)
    degrees_of_base_nodes = deg[base_nodes_idx]

    if degrees_of_base_nodes.sum() == 0:
        if num_base_nodes > 0:
            sampled_indices_in_base = torch.randperm(num_base_nodes, device=data.x.device)[:num_select]
            sampled_nodes = base_nodes_idx[sampled_indices_in_base]
        else:
            sampled_nodes = torch.tensor([], dtype=torch.long, device=data.x.device)
    else:
        probs = degrees_of_base_nodes.float() / degrees_of_base_nodes.sum().float()
        probs = torch.nan_to_num(probs, nan=1.0/num_base_nodes if num_base_nodes > 0 else 0.0)
        if probs.sum() == 0 and num_base_nodes > 0 :
             sampled_indices_in_base = torch.randperm(num_base_nodes, device=data.x.device)[:num_select]
             sampled_nodes = base_nodes_idx[sampled_indices_in_base]
        elif num_select > 0 and num_base_nodes > 0 and probs.sum() > 0:
             sampled_indices_in_base = torch.multinomial(probs, num_select, replacement=False)
             sampled_nodes = base_nodes_idx[sampled_indices_in_base]
        else:
             sampled_nodes = torch.tensor([], dtype=torch.long, device=data.x.device)

    output_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    if sampled_nodes.numel() > 0:
        output_mask[sampled_nodes] = True
    return output_mask


def soft_masking_gnn_input(x, gnn_perturb_mask, mask_token_embed, beta=0.7):
    device = x.device
    gnn_perturb_mask = gnn_perturb_mask.to(device)
    mask_token_embed = mask_token_embed.to(device)
    x_masked = x.clone()
    if gnn_perturb_mask.any():
        x_masked[gnn_perturb_mask] = (1 - beta) * x[gnn_perturb_mask] + beta * mask_token_embed
    return x_masked


def nt_xent_loss(z1, z2, temperature=0.5, batch_size=8):
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


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        B, N, C = x.shape
        H = self.num_heads

        q = self.q_proj(x).reshape(B, N, H, C // H).permute(0, 2, 1, 3)
        k = self.k_proj(y).reshape(B, N, H, C // H).permute(0, 2, 1, 3)
        v = self.v_proj(y).reshape(B, N, H, C // H).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x

class MultiScaleFusion(nn.Module):
    def __init__(self, hidden_dims, output_dim):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(len(hidden_dims)) / len(hidden_dims))
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in hidden_dims
        ])
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, embeddings_list):
        weights = F.softmax(self.scale_weights, dim=0)
        projected = [proj(emb) for proj, emb in zip(self.projections, embeddings_list)]
        weighted = sum(w * emb for w, emb in zip(weights, projected))
        return self.layer_norm(weighted)

class GraphTextLM(nn.Module):
    def __init__(self, gnn_in_channels, hidden_channels, num_classes, num_relations=5, num_bases=30, dropout_rate=0.3, model_name='thenlper/gte-base', plm_max_length=256):
        super(GraphTextLM, self).__init__()
        self.gnn_mask_token_embed = nn.Parameter(torch.zeros(1, gnn_in_channels))
        nn.init.xavier_uniform_(self.gnn_mask_token_embed)

        # RGCN layers
        self.rgcn1 = RGCNConv(gnn_in_channels, hidden_channels, num_relations=num_relations, num_bases=num_bases)
        self.gnorm1 = GraphNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.rgcn2 = RGCNConv(hidden_channels, hidden_channels * 2, num_relations=num_relations, num_bases=num_bases)
        self.gnorm2 = GraphNorm(hidden_channels * 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.rgcn3 = RGCNConv(hidden_channels * 2, hidden_channels * 4, num_relations=num_relations, num_bases=num_bases)
        self.gnorm3 = GraphNorm(hidden_channels * 4)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.rgcn4 = RGCNConv(hidden_channels * 4, hidden_channels * 8, num_relations=num_relations, num_bases=num_bases)
        self.gnorm4 = GraphNorm(hidden_channels * 8)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.residual_proj1 = nn.Linear(gnn_in_channels, hidden_channels)
        self.residual_proj2 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.residual_proj3 = nn.Linear(hidden_channels * 2, hidden_channels * 8)
        
        gnn_output_dim = hidden_channels * 8

        # PLM encoder
        logger.info(f"Loading HuggingFace model: {model_name}")
        self.plm_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.plm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.plm_max_length = plm_max_length
        # gradient checkpointing 
        if hasattr(self.plm_encoder, 'gradient_checkpointing_enable'):
            self.plm_encoder.gradient_checkpointing_enable()
            logger.info("Enabled PLM internal gradient checkpointing.")
            
        logger.info("HuggingFace model loaded.")
        plm_hidden_size = self.plm_encoder.config.hidden_size

        # multi-scale fusion for graph embeddings
        self.multi_scale_fusion = MultiScaleFusion(
            [hidden_channels, hidden_channels * 2, hidden_channels * 4, hidden_channels * 8],
            plm_hidden_size
        )
        
        # cross-attention layers
        self.graph_to_text_attn = CrossAttention(plm_hidden_size, num_heads=8, dropout=dropout_rate)
        self.text_to_graph_attn = CrossAttention(plm_hidden_size, num_heads=8, dropout=dropout_rate)
        
        # final fusion and classification
        fusion_hidden = plm_hidden_size
        self.fusion_network = nn.Sequential(
            nn.Linear(plm_hidden_size * 2, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, num_classes)
        )

    def get_graph_embeddings(self, x_feat, edge_index, edge_type=None):
        edge_index = edge_index.to(torch.long)
        
        if edge_type is None:
            num_edges = edge_index.size(1)
            edge_type = torch.zeros(num_edges, dtype=torch.long, device=edge_index.device)
            deg = degree(edge_index[0], num_nodes=x_feat.size(0))
            for i in range(num_edges):
                src_node = edge_index[0, i]
                src_degree = deg[src_node]
                if src_degree <= 2:
                    edge_type[i] = 0
                elif src_degree <= 5:
                    edge_type[i] = 1
                elif src_degree <= 10:
                    edge_type[i] = 2
                else:
                    edge_type[i] = 3
        
        embeddings_list = []
        
        def forward_rgcn1(x_feat, edge_index, edge_type):
            x1 = self.rgcn1(x_feat, edge_index, edge_type)
            if x1.size(0) > 1: x1 = self.gnorm1(x1)
            x1 = F.gelu(x1)
            x1 = self.dropout1(x1)
            return x1
        
        x1 = checkpoint(forward_rgcn1, x_feat, edge_index, edge_type, use_reentrant=False)
        embeddings_list.append(x1)
        
        residual1 = self.residual_proj1(x_feat)
        x1 = x1 + residual1
        
        def forward_rgcn2(x1, edge_index, edge_type):
            x2 = self.rgcn2(x1, edge_index, edge_type)
            if x2.size(0) > 1: x2 = self.gnorm2(x2)
            x2 = F.gelu(x2)
            x2 = self.dropout2(x2)
            return x2
        
        x2 = checkpoint(forward_rgcn2, x1, edge_index, edge_type, use_reentrant=False)
        embeddings_list.append(x2)
        
        residual2 = self.residual_proj2(x1)
        x2 = x2 + residual2
        
        def forward_rgcn3(x2, edge_index, edge_type):
            x3 = self.rgcn3(x2, edge_index, edge_type)
            if x3.size(0) > 1: x3 = self.gnorm3(x3)
            x3 = F.gelu(x3)
            x3 = self.dropout3(x3)
            return x3
        
        x3 = checkpoint(forward_rgcn3, x2, edge_index, edge_type, use_reentrant=False)
        embeddings_list.append(x3)
        
        def forward_rgcn4(x3, edge_index, edge_type):
            x4 = self.rgcn4(x3, edge_index, edge_type)
            if x4.size(0) > 1: x4 = self.gnorm4(x4)
            x4 = F.gelu(x4)
            x4 = self.dropout4(x4)
            return x4
        
        x4 = checkpoint(forward_rgcn4, x3, edge_index, edge_type, use_reentrant=False)
        embeddings_list.append(x4)
        
        residual3 = self.residual_proj3(x2)
        x4 = x4 + residual3
        
        return self.multi_scale_fusion(embeddings_list)

    def forward(self, gnn_input_features, edge_index, all_node_texts, text_processing_node_mask, edge_type=None, plm_batch_size=8):
        edge_index = edge_index.to(torch.long)
        num_nodes = gnn_input_features.size(0)

        gnn_embeds = self.get_graph_embeddings(gnn_input_features, edge_index, edge_type)
        
        plm_embeds = torch.zeros(num_nodes, self.plm_encoder.config.hidden_size,
                                device=gnn_input_features.device)

        active_node_indices = text_processing_node_mask.nonzero(as_tuple=True)[0]
        
        if active_node_indices.numel() > 0:
            for batch_start in range(0, active_node_indices.numel(), plm_batch_size):
                batch_end = batch_start + plm_batch_size
                batch_indices = active_node_indices[batch_start:batch_end]
                
                texts_to_encode = [all_node_texts[i.item()] for i in batch_indices]
                if not texts_to_encode:
                    continue
                
                encodings = self.plm_tokenizer.batch_encode_plus(
                    texts_to_encode, padding=True, truncation=True,
                    max_length=self.plm_max_length, return_tensors="pt"
                ).to(gnn_input_features.device)

                with torch.set_grad_enabled(self.plm_encoder.training or self.training):
                    with torch.amp.autocast('cuda'):
                        plm_output = self.plm_encoder(**encodings)
                        
                        attention_mask = encodings['attention_mask']
                        last_hidden = plm_output.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        batch_plm_embeds = sum_embeddings / sum_mask

                plm_embeds[batch_indices] = batch_plm_embeds

        gnn_embeds = gnn_embeds.unsqueeze(0)
        plm_embeds = plm_embeds.unsqueeze(0)

        gnn_attended = self.graph_to_text_attn(gnn_embeds, plm_embeds)
        text_attended = self.text_to_graph_attn(plm_embeds, gnn_embeds)
        
        fused_features = torch.cat([gnn_attended, text_attended], dim=-1)
        fused_representation = self.fusion_network(fused_features)
        
        fused_representation = fused_representation.squeeze(0)
        
        logits = self.classifier(fused_representation)
        return logits


def setup_optimizer(model, lr_graph, lr_bert, lr_other, weight_decay):
    graph_params = []
    bert_params = []
    other_params = []
    gnn_param_names = ['rgcn1', 'rgcn2', 'rgcn3', 'gnorm1', 'gnorm2', 'gnorm3', 'residual_proj1', 'residual_proj2', 'residual_proj3']
    plm_param_name = 'plm_encoder.'

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(plm_param_name):
            bert_params.append(param)
        elif any(gnn_layer_name in name for gnn_layer_name in gnn_param_names):
            graph_params.append(param)
        else:
            other_params.append(param)

    logger.info(f"Optimizer setup: GNN params={len(graph_params)}, PLM params={len(bert_params)}, Other params={len(other_params)}")
    optimizer = optim.AdamW([
        {'params': graph_params, 'lr': lr_graph, 'weight_decay': weight_decay},
        {'params': bert_params, 'lr': lr_bert, 'weight_decay': 0.01},
        {'params': other_params, 'lr': lr_other, 'weight_decay': weight_decay}
    ])
    return optimizer


def pretrain_contrastive_gnn(model, data, pretrain_epochs=20, temp=0.5,
                            gnn_perturb_mask_ratio_min=0.2, gnn_perturb_mask_ratio_max=0.4, beta_soft_mask=0.7,
                            lr_graph=1e-3, lr_other_pretrain=1e-4, weight_decay=0.01, verbose=True):
    logger.info("Starting GNN contrastive pretraining...")
    
    original_param_requires_grad = {}
    for name, param in model.named_parameters():
        original_param_requires_grad[name] = param.requires_grad
        if name.startswith(model.plm_encoder.base_model_prefix) or \
           name.startswith('fusion_network') or \
           name.startswith('classifier') or \
           name.startswith('gnn_proj'):
            param.requires_grad = False
        elif any(gnn_layer_name in name for gnn_layer_name in ['rgcn1', 'rgcn2', 'rgcn3', 'gnorm1', 'gnorm2', 'gnorm3', 'residual_proj1', 'residual_proj2', 'residual_proj3']) or \
             'gnn_mask_token_embed' in name :
            param.requires_grad = True

    gnn_pretrain_params = [p for n, p in model.named_parameters() if p.requires_grad]
    if not gnn_pretrain_params:
        logger.warning("No parameters to pretrain for GNN. Skipping pretraining.")
        for name, param in model.named_parameters():
            param.requires_grad = original_param_requires_grad[name]
        return 0.0

    optimizer = optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if any(gl in n for gl in ['rgcn1', 'rgcn2', 'rgcn3', 'gnorm1', 'gnorm2', 'gnorm3', 'residual_proj1', 'residual_proj2', 'residual_proj3'])], 'lr': lr_graph, 'weight_decay': weight_decay},
        {'params': [model.gnn_mask_token_embed], 'lr': lr_other_pretrain, 'weight_decay': weight_decay},
    ])
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    losses = []
    model.train()

    pbar = trange(pretrain_epochs, desc="GNN Pretraining", leave=False, 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    
    start_time = time.time()
    for epoch in pbar:
        optimizer.zero_grad()
        gnn_perturb_mask1 = generate_active_node_mask(data, random.uniform(gnn_perturb_mask_ratio_min, gnn_perturb_mask_ratio_max), base_mask_name=None)
        gnn_perturb_mask2 = generate_active_node_mask(data, random.uniform(gnn_perturb_mask_ratio_min, gnn_perturb_mask_ratio_max), base_mask_name=None)

        x1_perturbed = soft_masking_gnn_input(data.x, gnn_perturb_mask1, model.gnn_mask_token_embed, beta=beta_soft_mask)
        x2_perturbed = soft_masking_gnn_input(data.x, gnn_perturb_mask2, model.gnn_mask_token_embed, beta=beta_soft_mask)

        with torch.amp.autocast('cuda'):
            g1 = model.get_graph_embeddings(x1_perturbed, data.edge_index, edge_type=None)
            g2 = model.get_graph_embeddings(x2_perturbed, data.edge_index, edge_type=None)
            loss = nt_xent_loss(g1, g2, temperature=temp, batch_size=8)

        if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch)
            losses.append(loss.item())
            
            current_lr_g = optimizer.param_groups[0]['lr']
            current_lr_o = optimizer.param_groups[1]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR_G': f'{current_lr_g:.1e}',
                'LR_O': f'{current_lr_o:.1e}'
            })
        else:
             logger.warning(f"Invalid loss encountered (Epoch {epoch}): {loss}. Skipping step.")
             losses.append(np.nan)
             pbar.set_postfix({'Loss': 'NaN', 'Status': 'Skip'})

        if verbose and (epoch % 10 == 0 or epoch == pretrain_epochs - 1):
            current_lr_g = optimizer.param_groups[0]['lr']
            current_lr_o = optimizer.param_groups[1]['lr']
            logger.info(f"Pretrain GNN Epoch: {epoch}, Loss: {loss.item():.4f}, LRs: G={current_lr_g:.1e}, O={current_lr_o:.1e}")

    pbar.close()
    
    for name, param in model.named_parameters():
        param.requires_grad = original_param_requires_grad[name]

    avg_loss = np.nanmean(losses[-5:]) if losses else 0
    elapsed_time = time.time() - start_time
    
    if verbose:
        logger.info(f"GNN Pretraining completed in {elapsed_time:.2f}s. Final Avg Loss (last 5 epochs): {avg_loss:.4f}")
    return avg_loss


def train_model(model, data, num_epochs=500,
                active_node_mask_ratio_min=0.2, active_node_mask_ratio_max=0.4,
                beta_soft_mask_gnn=0.7,
                lr_graph=1e-3, lr_bert=1e-5, lr_other=1e-4, weight_decay=0.01,
                patience=20, warmup_ratio=0.1, grad_clip_norm=1.0, verbose=True,
                plm_batch_size=32, model_params=None, train_params=None):
    logger.info("Starting model training...")
    model.train()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = setup_optimizer(model, lr_graph, lr_bert, lr_other, weight_decay)

    scaler = torch.amp.GradScaler('cuda')
    
    num_training_steps = num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    if WANDB_AVAILABLE and verbose and model_params is not None and train_params is not None:
        wandb.init(project="GraphTextLM-RGCN-Improved", config={**model_params, **train_params})
        wandb.watch(model)

    losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_model_state_dict = None
    stopped_epoch = num_epochs

    use_early_stopping = hasattr(data, 'val_mask') and data.val_mask is not None and data.val_mask.any()
    if not use_early_stopping:
        logger.warning("No valid validation mask found. Early stopping disabled.")

    pbar = trange(num_epochs, desc="Training", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    
    start_time = time.time()
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        train_active_nodes_mask = generate_active_node_mask(data,
                                                            random.uniform(active_node_mask_ratio_min, active_node_mask_ratio_max),
                                                            base_mask_name='train_mask')

        if not train_active_nodes_mask.any():
             logger.warning(f"Epoch {epoch}: No training nodes selected by active_mask. Skipping epoch.")
             pbar.set_postfix({'Status': 'No nodes', 'Loss': 'N/A'})
             continue

        gnn_input_feat_perturbed = soft_masking_gnn_input(data.x, train_active_nodes_mask, model.gnn_mask_token_embed, beta=beta_soft_mask_gnn)

        with torch.amp.autocast('cuda'):
            # OPTIMIZATION: Pass plm_batch_size to the model's forward pass
            logits = model(gnn_input_feat_perturbed, data.edge_index, data.node_texts, train_active_nodes_mask, edge_type=None, plm_batch_size=plm_batch_size)
            if data.y[train_active_nodes_mask].numel() == 0:
                logger.warning(f"Epoch {epoch}: No labels available for the selected active_mask. Skipping loss computation.")
                pbar.set_postfix({'Status': 'No labels', 'Loss': 'N/A'})
                continue
            loss = criterion(logits[train_active_nodes_mask], data.y[train_active_nodes_mask])
        
        with torch.no_grad():
            train_pred = logits[train_active_nodes_mask].max(dim=1)[1]
            train_acc = train_pred.eq(data.y[train_active_nodes_mask]).sum().item() / train_active_nodes_mask.sum().item()
            train_accuracies.append(train_acc)

        if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Train_Acc': f'{train_acc*100:.1f}%',
                'Best_Val_F1': f'{best_val_f1:.4f}',
                'Patience': f'{epochs_no_improve}/{patience}'
            })
        else:
            logger.warning(f"Invalid training loss encountered (Epoch {epoch}): {loss}. Skipping step.")
            losses.append(np.nan)
            pbar.set_postfix({'Loss': 'NaN', 'Status': 'Skip'})
            continue

        if verbose and (epoch % 20 == 0 or epoch == num_epochs - 1):
             current_lr_g = optimizer.param_groups[0]['lr']
             current_lr_b = optimizer.param_groups[1]['lr']
             current_lr_o = optimizer.param_groups[2]['lr']
             logger.info(f"Train Epoch: {epoch}, Loss: {loss.item():.4f}, Train Acc: {train_acc*100:.2f}%, LRs: G={current_lr_g:.1e}, B={current_lr_b:.1e}, O={current_lr_o:.1e}")

        if use_early_stopping and (epoch % 5 == 0 or epoch == num_epochs - 1):
            model.eval()
            with torch.no_grad():
                val_mask = data.val_mask
                
                if val_mask.dim() > 1:
                    if val_mask.shape[1] == 1:
                        val_mask = val_mask.squeeze(1)
                    else:
                        val_mask = val_mask.any(dim=1)
                
                if val_mask.dtype != torch.bool:
                    val_mask = val_mask.bool()
                    
                if not val_mask.any():
                    logger.warning(f"Epoch: {epoch}, No validation nodes found, skipping validation")
                    continue
                
                # OPTIMIZATION: Pass plm_batch_size during validation
                val_logits = model(data.x, data.edge_index, data.node_texts, val_mask, edge_type=None, plm_batch_size=plm_batch_size)
                
                val_labels = data.y[val_mask]
                val_predictions = val_logits[val_mask]
                
                val_loss = eval_criterion(val_predictions, val_labels)
                val_pred = val_predictions.max(dim=1)[1]
                val_acc = val_pred.eq(val_labels).sum().item() / val_mask.sum().item()
                val_f1 = calculate_f1(val_pred.cpu().numpy(), val_labels.cpu().numpy())
                
                val_losses.append(val_loss.item())
                val_accuracies.append(val_acc)
                val_f1s.append(val_f1)

            if verbose:
                logger.info(f"Validate Epoch: {epoch}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                best_model_state_dict = copy.deepcopy(model.state_dict())
                if verbose:
                    logger.info(f"     New best validation F1: {best_val_f1:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} after {patience} epochs with no improvement on Val F1.")
                stopped_epoch = epoch
                break

    pbar.close()
    
    avg_loss = np.nanmean(losses[-5:]) if losses else 0
    elapsed_time = time.time() - start_time
    
    if verbose:
        logger.info(f"Training completed in {elapsed_time:.2f}s. Final Avg Training Loss (last 5 epochs): {avg_loss:.4f}")

    if use_early_stopping and best_model_state_dict is not None:
        logger.info(f"Loading best model state from epoch {stopped_epoch - patience if stopped_epoch > patience else 0} with Val F1: {best_val_f1:.4f}")
        model.load_state_dict(best_model_state_dict)
    elif not use_early_stopping:
         logger.info("Training completed full epochs without early stopping.")
    else:
         logger.warning("Early stopping enabled but no best model state was loaded (e.g. validation never improved). Using last model state.")

    if losses:
        logger.info(f"Training Summary - Total epochs: {len(losses)}, Avg loss: {np.mean(losses):.4f}, Final loss: {losses[-1]:.4f}")
    if train_accuracies:
        logger.info(f"Training Accuracy Summary - Avg: {np.mean(train_accuracies)*100:.2f}%, Final: {train_accuracies[-1]*100:.2f}%")
    if val_f1s:
        logger.info(f"Validation Summary - Best F1: {max(val_f1s):.4f}, Final F1: {val_f1s[-1]:.4f}")

    if WANDB_AVAILABLE and verbose:
        wandb.log({
            'train/loss': avg_loss,
            'train/acc': np.mean(train_accuracies) * 100,
            'epoch': num_epochs
        })

    if WANDB_AVAILABLE and verbose:
        wandb.finish()
    return avg_loss


def evaluate_model(model, data, mask_name='test_mask', plm_batch_size=32):
    logger.info(f"Starting evaluation on {mask_name}...")
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    if not hasattr(data, mask_name) or getattr(data, mask_name) is None:
        logger.error(f"No '{mask_name}' found in data. Cannot evaluate.")
        return {'loss': float('nan'), 'accuracy': 0.0, 'f1': 0.0}

    eval_mask = getattr(data, mask_name)
    
    if eval_mask.dim() > 1:
        if eval_mask.shape[1] == 1:
            eval_mask = eval_mask.squeeze(1)
        else:
            eval_mask = eval_mask.any(dim=1)
    
    if eval_mask.dtype != torch.bool:
        eval_mask = eval_mask.bool()
        
    if not eval_mask.any():
        logger.error(f"No valid nodes in '{mask_name}'. Cannot evaluate.")
        return {'loss': float('nan'), 'accuracy': 0.0, 'f1': 0.0}

    start_time = time.time()
    with torch.no_grad():
        # OPTIMIZATION: Pass plm_batch_size during evaluation
        logits = model(data.x, data.edge_index, data.node_texts, eval_mask, edge_type=None, plm_batch_size=plm_batch_size)

        eval_labels = data.y[eval_mask]
        eval_predictions = logits[eval_mask]
        
        if eval_labels.numel() == 0:
            logger.warning(f"No labels available for nodes in '{mask_name}'.")
            return {'loss': float('nan'), 'accuracy': 0.0, 'f1': 0.0}

        loss = criterion(eval_predictions, eval_labels)
        pred = eval_predictions.max(dim=1)[1]
        acc = pred.eq(eval_labels).sum().item() / eval_mask.sum().item()
        f1 = calculate_f1(pred.cpu().numpy(), eval_labels.cpu().numpy())

        eval_time = time.time() - start_time
        
        logger.info(f"Evaluation on '{mask_name}' completed in {eval_time:.2f}s")
        logger.info(f"   Results - Loss: {loss.item():.4f}, Accuracy: {acc*100:.2f}%, F1: {f1:.4f}")
        logger.info(f"   Samples evaluated: {eval_mask.sum().item()}")
        
        return {
            'loss': loss.item(),
            'accuracy': acc * 100,
            'f1': f1
        }


def calculate_f1(y_pred, y_true):
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def load_texts_from_jsonl(jsonl_path):
    texts = []
    if not os.path.exists(jsonl_path):
        logger.error(f"JSONL file not found: {jsonl_path}")
        return texts
        
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    text = None
                    for field in ['text', 'content', 'body', 'description', 'title', 'page_content']:
                        if field in data and data[field]:
                            text = str(data[field]).strip()
                            break
                    
                    if text:
                        texts.append(text)
                    else:
                        text_parts = []
                        for key, value in data.items():
                            if isinstance(value, str) and value.strip():
                                text_parts.append(f"{key}: {value.strip()}")
                        if text_parts:
                            texts.append(" | ".join(text_parts))
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num} in {jsonl_path}: {e}")
                    continue
                    
        logger.info(f"Loaded {len(texts)} texts from {jsonl_path}")
        return texts
        
    except Exception as e:
        logger.error(f"Error reading JSONL file {jsonl_path}: {e}")
        return texts

def sample_texts_for_nodes(texts, num_nodes, dataset_name):
    if not texts:
        logger.error(f"No texts available for {dataset_name}. Using fallback text.")
        return [f"University web page content for {dataset_name}" for _ in range(num_nodes)]
    
    if len(texts) >= num_nodes:
        sampled_texts = random.sample(texts, num_nodes)
        logger.info(f"Sampled {num_nodes} texts from {len(texts)} available texts for {dataset_name}")
    else:
        sampled_texts = random.choices(texts, k=num_nodes)
        logger.warning(f"Only {len(texts)} texts available for {num_nodes} nodes in {dataset_name}. Sampling with replacement.")
    
    return sampled_texts

def load_npz_dataset(dataset_name, npz_path, split_ratios=None, seed=42):
    logger.info(f"Loading {dataset_name} from {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    x = torch.tensor(d['node_features'], dtype=torch.float)
    edge_index = torch.tensor(d['edges'], dtype=torch.long)
    y = torch.tensor(d['node_labels'], dtype=torch.long)
    node_texts = list(d['node_texts'])
    label_texts = list(d['label_texts'])
    num_nodes = x.size(0)
    num_classes = len(set(y.tolist()))
    num_features = x.size(1)

    if split_ratios is not None:
        train_ratio, val_ratio, test_ratio = split_ratios
        idx = np.arange(num_nodes)
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        n_train = int(train_ratio * num_nodes)
        n_val = int(val_ratio * num_nodes)
        n_test = num_nodes - n_train - n_val
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train+n_val:]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    else:
        train_mask = torch.tensor(d['train_masks'], dtype=torch.bool)
        val_mask = torch.tensor(d['val_masks'], dtype=torch.bool)
        test_mask = torch.tensor(d['test_masks'], dtype=torch.bool)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.node_texts = node_texts
    data.label_texts = label_texts
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data, num_features, num_classes


def augment_texts(texts):
    synonym_dict = {'university': 'college', 'student': 'learner', 'research': 'study', 'professor': 'instructor'}
    augmented = []
    for t in texts:
        for k, v in synonym_dict.items():
            t = t.replace(k, v)
        augmented.append(t)
    return augmented

def augment_graph(data, edge_dropout_p=0.1):
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > edge_dropout_p
    data.edge_index = edge_index[:, keep_mask]
    return data

def load_dataset_with_texts(dataset_name, root_path, split_transform_flag):
    npz_map = {
        'Cornell': ('collapse/data/Cornell.npz', (0.48, 0.32, 0.20)),
        'Texas': ('collapse/data/Texas.npz', (0.48, 0.32, 0.20)),
        'Wisconsin': ('collapse/data/Wisconsin.npz', (0.48, 0.32, 0.20)),
        'Actor': ('collapse/data/Actor.npz', (0.48, 0.32, 0.20)),
        'Amazon': ('collapse/data/Amazon.npz', (0.50, 0.25, 0.25)),
    }
    if dataset_name not in npz_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    npz_path, split = npz_map[dataset_name]
    data, num_features, num_classes = load_npz_dataset(dataset_name, npz_path, split_ratios=split)
    data.node_texts = augment_texts(data.node_texts)
    data = augment_graph(data, edge_dropout_p=0.1)
    logger.info(f"{dataset_name} loaded: nodes={data.x.size(0)}, features={num_features}, classes={num_classes}")
    logger.info(f"   Splits: train={data.train_mask.sum().item()}, val={data.val_mask.sum().item()}, test={data.test_mask.sum().item()}")
    logger.info(f"   Sample text: {data.node_texts[0][:100] if data.node_texts else 'N/A'}")
    return None, data


def run_multiple_experiments(dataset_name, dataset, data, model_params, train_params, num_runs=10, verbose=True):
    logger.info(f"Starting {num_runs} experiments for {dataset_name}")
    
    all_results = []
    gnn_in_channels = data.x.size(1)
    num_classes = len(torch.unique(data.y))
    
    run_pbar = trange(num_runs, desc=f"{dataset_name} Experiments", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    
    successful_runs = 0
    start_time = time.time()
    
    for run in run_pbar:
        run_start_time = time.time()
        logger.info(f"Run {run + 1}/{num_runs} for {dataset_name}")
        
        model = GraphTextLM(
            gnn_in_channels, 
            model_params['hidden_channels'], 
            num_classes,
            model_params['num_relations'],
            model_params['num_bases'],
            model_params['dropout_rate'],
            model_params['plm_model_name'], 
            model_params['plm_max_length']
        ).to(device)
        
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        random.seed(42 + run)
        
        if train_params['apply_split']:
            logger.info(f"Applying 60/20/20 split for run {run + 1}")
            transform = T.RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
            data_run = transform(copy.deepcopy(data))
        else:
            data_run = data
        
        try:
            logger.info(f"Starting GNN contrastive pretraining for run {run + 1}...")
            pretrain_contrastive_gnn(
                model, data_run, 
                pretrain_epochs=train_params['pretrain_epochs'],
                temp=train_params['temp_contrastive'],
                gnn_perturb_mask_ratio_min=0.2, 
                gnn_perturb_mask_ratio_max=0.5,
                beta_soft_mask=train_params['beta_soft_mask'], 
                verbose=False,
                lr_graph=train_params['lr_graph'], 
                lr_other_pretrain=train_params['lr_other'], 
                weight_decay=train_params['weight_decay']
            )
            
            torch.cuda.empty_cache()
            
            logger.info(f"Starting fine-tuning for run {run + 1}...")
            train_model(
                model, data_run, 
                num_epochs=train_params['train_epochs'],
                active_node_mask_ratio_min=train_params['active_node_mask_ratio_min'],
                active_node_mask_ratio_max=train_params['active_node_mask_ratio_max'],
                beta_soft_mask_gnn=train_params['beta_soft_mask'], 
                verbose=False,
                lr_graph=train_params['lr_graph'], 
                lr_bert=train_params['lr_bert'], 
                lr_other=train_params['lr_other'],
                weight_decay=train_params['weight_decay'], 
                patience=train_params['patience'],
                warmup_ratio=train_params['warmup_ratio'],
                grad_clip_norm=train_params['grad_clip_norm'],
                plm_batch_size=train_params['plm_batch_size'],
                model_params=model_params,
                train_params=train_params
            )
            
            torch.cuda.empty_cache()
            
            logger.info(f"Evaluating run {run + 1}...")
            eval_metrics = evaluate_model(model, data_run, mask_name='test_mask', plm_batch_size=train_params['plm_batch_size'])
            all_results.append(eval_metrics)
            successful_runs += 1
            
            run_time = time.time() - run_start_time
            logger.info(f"Run {run + 1} completed in {run_time:.2f}s - Acc: {eval_metrics['accuracy']:.2f}%, F1: {eval_metrics['f1']:.4f}")
            
            run_pbar.set_postfix({
                'Success': f'{successful_runs}/{run+1}',
                'Last_Acc': f"{eval_metrics['accuracy']:.1f}%",
                'Last_F1': f"{eval_metrics['f1']:.3f}"
            })
            
        except Exception as e:
            logger.error(f"Error in run {run + 1}: {e}", exc_info=True) 
            all_results.append({'loss': float('nan'), 'accuracy': 0.0, 'f1': 0.0})
            run_pbar.set_postfix({
                'Success': f'{successful_runs}/{run+1}',
                'Status': 'Failed'
            })
    
    run_pbar.close()
    
    valid_results = [r for r in all_results if not np.isnan(r['loss']) and r['accuracy'] > 0]
    total_time = time.time() - start_time
    
    if len(valid_results) == 0:
        logger.warning(f"No successful runs for {dataset_name}")
        return {'loss_mean': float('nan'), 'loss_std': float('nan'), 
                'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                'f1_mean': 0.0, 'f1_std': 0.0, 'num_successful_runs': 0}
    
    accuracy_values = [r['accuracy'] for r in valid_results]
    f1_values = [r['f1'] for r in valid_results]
    loss_values = [r['loss'] for r in valid_results if not np.isnan(r['loss'])]
    
    results_summary = {
        'loss_mean': np.mean(loss_values) if loss_values else float('nan'),
        'loss_std': np.std(loss_values) if loss_values else float('nan'),
        'accuracy_mean': np.mean(accuracy_values),
        'accuracy_std': np.std(accuracy_values),
        'f1_mean': np.mean(f1_values),
        'f1_std': np.std(f1_values),
        'num_successful_runs': len(valid_results),
        'all_results': all_results
    }
    
    logger.info(f"{dataset_name} Summary ({len(valid_results)}/{num_runs} successful runs, {total_time:.2f}s total)")
    logger.info(f"   Accuracy: {results_summary['accuracy_mean']:.2f}% ± {results_summary['accuracy_std']:.2f}%")
    logger.info(f"   F1 Score: {results_summary['f1_mean']:.4f} ± {results_summary['f1_std']:.4f}")
    
    return results_summary


def main():
    verbose = True
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    logger.info("Starting GraphTextLM Experiments")
    logger.info(f"   Device: {device}")
    logger.info(f"   Results will be saved to: ./results/")
    logger.info(f"   Training logs saved to: training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    model_params = {
        'hidden_channels': 512,
        'num_relations': 5,
        'num_bases': 30,
        'dropout_rate': 0.5,
        'plm_model_name': 'Qwen/Qwen3-Embedding-0.6B',  
        'plm_max_length': 512
    }
    
    train_params = {
        'beta_soft_mask': 0.7,
        'active_node_mask_ratio_min': 0.3,
        'active_node_mask_ratio_max': 0.8,
        'lr_graph': 1e-4,
        'temp_contrastive': 0.1,
        'pretrain_epochs': 30,
        'train_epochs': 500,
        'patience': 30,
        'lr_bert': 1e-5,
        'lr_other': 1e-4,
        'weight_decay': 0.05,
        'warmup_ratio': 0.1,
        'grad_clip_norm': 1.0,
        'apply_split': False,
        # micro-batch size for PLM. Adjust based on VRAM.
        'plm_batch_size': 32 
    }

    logger.info("Model Parameters:")
    for key, value in model_params.items():
        logger.info(f"   • {key}: {value}")
    
    logger.info("Training Parameters:")
    for key, value in train_params.items():
        logger.info(f"   • {key}: {value}")

    dataset_configs_list = [
        {'name': 'Cornell', 'split_transform': False},
        {'name': 'Texas', 'split_transform': False},
        {'name': 'Wisconsin', 'split_transform': False},
        {'name': 'Actor', 'split_transform': False},
        {'name': 'Amazon', 'split_transform': False},
    ]

    logger.info(f"Processing {len(dataset_configs_list)} datasets: {[config['name'] for config in dataset_configs_list]}")
    
    all_final_results = {}
    experiment_start_time = time.time()

    dataset_pbar = tqdm(dataset_configs_list, desc="Datasets", 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
    
    for config in dataset_pbar:
        dataset_name = config['name']
        
        dataset_pbar.set_description(f"Processing {dataset_name}")
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING DATASET: {dataset_name}")
        logger.info(f"{'='*60}")

        try:
            dataset, data = load_dataset_with_texts(dataset_name, root_path=None, split_transform_flag=False)
            if data is None:
                logger.error(f"Skipping {dataset_name} due to loading errors.")
                dataset_pbar.set_description(f"{dataset_name} - Failed")
                continue
            data = data.to(device)

            if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y') or not hasattr(data, 'node_texts'):
                 logger.error(f"Loaded data for {dataset_name} is missing required attributes. Skipping.")
                 dataset_pbar.set_description(f"{dataset_name} - Invalid")
                 continue

            train_params['apply_split'] = False
            
            logger.info(f"Starting 10-run experiments for {dataset_name}")
            results_summary = run_multiple_experiments(
                dataset_name, None, data, model_params, train_params, num_runs=10, verbose=verbose
            )
            
            all_final_results[dataset_name] = results_summary
            
            if results_summary['num_successful_runs'] > 0:
                dataset_pbar.set_description(f"{dataset_name} - Acc: {results_summary['accuracy_mean']:.1f}%")
                logger.info(f"{dataset_name} completed successfully!")
            else:
                dataset_pbar.set_description(f"{dataset_name} - Failed")
                logger.error(f"{dataset_name} failed all runs!")
                
        except Exception as e:
            logger.error(f"Critical error processing {dataset_name}: {e}", exc_info=True)
            dataset_pbar.set_description(f"{dataset_name} - Error")
            continue

    dataset_pbar.close()
    
    total_experiment_time = time.time() - experiment_start_time
    
    results_filename = f'results/averaged_results_rgcn_10runs_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    try:
        json_results = {}
        for dataset_name, results in all_final_results.items():
            json_results[dataset_name] = {
                k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                for k, v in results.items() if k != 'all_results'
            }
        with open(results_filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Averaged results across 10 runs saved to {results_filename}")
    except Exception as e:
        logger.error(f"Error saving averaged results: {e}")

    logger.info(f"\n{'='*80}")
    logger.info("FINAL AVERAGED RESULTS SUMMARY (10 runs each)")
    logger.info(f"Total experiment time: {total_experiment_time:.2f}s ({total_experiment_time/3600:.2f}h)")
    logger.info(f"{'='*80}")
    
    summary_table = f"{'Dataset':<20} {'Accuracy (%)':<15} {'F1 Score':<15} {'Success Rate':<15}\n"
    summary_table += f"{'-'*80}\n"
    for dataset_name, results in all_final_results.items():
        acc_mean = results['accuracy_mean']
        acc_std = results['accuracy_std'] 
        f1_mean = results['f1_mean']
        f1_std = results['f1_std']
        success_rate = results['num_successful_runs']
        summary_table += f"{dataset_name:<20} {acc_mean:.2f}±{acc_std:.2f}    {f1_mean:.4f}±{f1_std:.4f}    {success_rate}/10\n"
    logger.info(f"\n{summary_table}")
    logger.info(f"{'='*80}")

    successful_datasets = [name for name, results in all_final_results.items() if results['num_successful_runs'] > 0]
    if successful_datasets:
        avg_accuracies = [results['accuracy_mean'] for results in all_final_results.values() if results['num_successful_runs'] > 0]
        avg_f1s = [results['f1_mean'] for results in all_final_results.values() if results['num_successful_runs'] > 0]
        logger.info(f"Overall Statistics:")
        logger.info(f"   Successful datasets: {len(successful_datasets)}/{len(dataset_configs_list)}")
        logger.info(f"   Average accuracy across datasets: {np.mean(avg_accuracies):.2f}% ± {np.std(avg_accuracies):.2f}%")
        logger.info(f"   Average F1 across datasets: {np.mean(avg_f1s):.4f} ± {np.std(avg_f1s):.4f}")
        logger.info(f"   Best performing dataset: {max(all_final_results.items(), key=lambda x: x[1]['f1_mean'])[0]}")

    logger.info(f"\nAll dataset processing completed with RGCN!")
    logger.info(f"Total runtime: {total_experiment_time:.2f}s ({total_experiment_time/3600:.2f}h)")
    logger.info(f"Results saved to: {results_filename}")
    logger.info(f"Full logs available in: training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    if WANDB_AVAILABLE and verbose:
        wandb.finish()

if __name__ == "__main__":
    main()