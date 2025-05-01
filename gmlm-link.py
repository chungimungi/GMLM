import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import degree, negative_sampling, train_test_split_edges
from transformers import AutoModel, AutoTokenizer
import random
import numpy as np
import os
from datetime import datetime
import time
import copy
import traceback
from tqdm.auto import tqdm
import logging
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

@torch.jit.script
def fast_degree_calculation(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row = edge_index[0]
    return degree(row, num_nodes=num_nodes) + 1

def generate_semantic_mask(data, mask_ratio):
    num_nodes = data.num_nodes
    num_mask = max(1, int(mask_ratio * num_nodes))
    dev = data.x.device if hasattr(data, 'x') and data.x is not None else device
    if num_nodes == 0:
        logger.debug("generate_semantic_mask: num_nodes is 0, returning empty mask.")
        return torch.zeros(num_nodes, dtype=torch.bool, device=dev)

    edge_index_for_deg = getattr(data, 'train_pos_edge_index', getattr(data, 'edge_index', None))
    has_edges = edge_index_for_deg is not None and edge_index_for_deg.numel() > 0

    if not has_edges:
        logger.debug("generate_semantic_mask: No edges found for degree calculation. Using random sampling.")
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
        if num_mask > 0 and num_nodes > 0:
            sampled_indices = torch.randperm(num_nodes, device=dev)[:num_mask]
            mask[sampled_indices] = True
        return mask

    deg = fast_degree_calculation(edge_index_for_deg, num_nodes)
    deg = deg.to(dev)
    deg_sum = deg.sum().float()

    if deg_sum == 0 or torch.isnan(deg_sum) or num_nodes == 0:
        logger.debug("generate_semantic_mask: Degree sum is zero or NaN. Using uniform probability.")
        probs = torch.ones(num_nodes, device=dev) / num_nodes if num_nodes > 0 else torch.tensor([], device=dev)
    else:
        probs = deg.float() / deg_sum

    probs = torch.nan_to_num(probs, nan=1.0/num_nodes if num_nodes > 0 else 0.0)
    if probs.sum() <= 0 and num_nodes > 0:
        logger.debug("generate_semantic_mask: Probability sum is zero. Using uniform probability.")
        probs = torch.ones(num_nodes, device=dev) / num_nodes

    num_mask = min(num_mask, num_nodes)
    logger.debug(f"generate_semantic_mask: num_nodes={num_nodes}, mask_ratio={mask_ratio:.2f}, num_mask={num_mask}")

    if num_nodes > 0 and num_mask > 0:
        try:
            probs = torch.clamp(probs, min=1e-9)
            prob_sum = probs.sum()
            if not torch.isclose(prob_sum, torch.tensor(1.0, device=probs.device), atol=1e-5):
                if prob_sum > 0:
                    probs = probs / prob_sum
                elif num_nodes > 0:
                    probs = torch.ones_like(probs) / num_nodes

            non_zero_probs_count = (probs > 0).sum().item()
            current_num_mask = min(num_mask, non_zero_probs_count)
            if current_num_mask != num_mask:
                logger.debug(f"generate_semantic_mask: Adjusted num_mask from {num_mask} to {current_num_mask} due to zero probabilities.")
                num_mask = current_num_mask

            if num_mask > 0 :
                if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() <= 0 or not torch.all(probs >= 0):
                    logger.warning("generate_semantic_mask: Invalid probabilities detected (NaN/inf/non-positive/negative). Falling back to random sampling.")
                    sampled_indices = torch.randperm(num_nodes, device=probs.device)[:num_mask]
                else:
                    sampled_indices = torch.multinomial(probs, num_mask, replacement=False)
            else:
                sampled_indices = torch.tensor([], dtype=torch.long, device=probs.device)

        except RuntimeError as e:
            logger.warning(f"generate_semantic_mask: Multinomial sampling failed ('{e}'). Falling back to random sampling.")
            sampled_indices = torch.randperm(num_nodes, device=probs.device)[:num_mask]

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=probs.device)
        if sampled_indices.numel() > 0:
             valid_indices = sampled_indices[sampled_indices < num_nodes]
             mask[valid_indices] = True
    else:
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=dev)

    logger.debug(f"generate_semantic_mask: Generated mask with {mask.sum().item()} nodes masked.")
    return mask

def soft_masking(x, mask, mask_token, beta=0.7):
    mask = mask.to(x.device)
    mask_token_on_device = mask_token.to(x.device)

    if mask_token_on_device.shape[0] != x.shape[1]:
        target_dim = x.shape[1]
        current_dim = mask_token_on_device.shape[0]
        logger.debug(f"soft_masking: Resizing mask token from dim {current_dim} to {target_dim}")
        if target_dim > current_dim:
            padding = torch.zeros(target_dim - current_dim, device=x.device)
            mask_token_resized = torch.cat([mask_token_on_device, padding])
        else:
            mask_token_resized = mask_token_on_device[:target_dim]
        mask_token_expanded = mask_token_resized.unsqueeze(0)
    else:
        mask_token_expanded = mask_token_on_device.unsqueeze(0)

    x_masked = x.clone()
    if mask.any():
        if mask.shape[0] == x.shape[0]:
            x_masked[mask] = (1 - beta) * x[mask] + beta * mask_token_expanded.expand_as(x[mask])
            logger.debug(f"soft_masking: Applied soft masking to {mask.sum().item()} nodes with beta={beta:.2f}")
        else:
             logger.warning(f"soft_masking: Mask shape {mask.shape} does not match input shape {x.shape}. Skipping masking.")

    return x_masked

def nt_xent_loss(z1, z2, temperature=0.5, batch_size=None):
    device = z1.device
    total_loss = 0.0
    num_batches = 0
    total_samples = z1.size(0)

    if total_samples <= 1:
        logger.debug("nt_xent_loss: Not enough samples to compute loss (<= 1). Returning 0.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    effective_batch_size = min(batch_size, total_samples) if batch_size else total_samples
    logger.debug(f"nt_xent_loss: total_samples={total_samples}, effective_batch_size={effective_batch_size}")

    for i in range(0, total_samples, effective_batch_size):
        end_idx = min(i + effective_batch_size, total_samples)
        batch_size_curr = end_idx - i

        if batch_size_curr <= 1:
            logger.debug(f"nt_xent_loss: Skipping batch {i // effective_batch_size} due to insufficient size ({batch_size_curr}).")
            continue

        batch_z1 = F.normalize(z1[i:end_idx], dim=1)
        batch_z2 = F.normalize(z2[i:end_idx], dim=1)

        batch_emb = torch.cat([batch_z1, batch_z2], dim=0)

        similarity = torch.mm(batch_emb.float(), batch_emb.t().float()) / temperature

        mask = torch.eye(2 * batch_size_curr, dtype=torch.bool, device=device)
        similarity_neg_filled = similarity.masked_fill(mask, -float('inf'))

        pos_indices = torch.arange(batch_size_curr, device=device)
        labels = torch.cat([pos_indices + batch_size_curr, pos_indices], dim=0)

        try:
            batch_loss = F.cross_entropy(similarity_neg_filled, labels, reduction='mean')
            if torch.isnan(batch_loss):
                logger.warning(f"nt_xent_loss: NaN loss detected in batch {i // effective_batch_size}. Skipping batch.")
                continue
            total_loss += batch_loss
            num_batches += 1
        except Exception as e:
            logger.error(f"nt_xent_loss: Error calculating cross entropy for batch {i // effective_batch_size}: {e}")
            continue

    if num_batches == 0:
        logger.warning("nt_xent_loss: No valid batches processed. Returning 0 loss.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    final_loss = total_loss / num_batches
    logger.debug(f"nt_xent_loss: Computed loss {final_loss:.4f} over {num_batches} batches.")
    return final_loss

class GraphLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nhead=4, model_name='distilbert/distilbert-base-uncased'):
        super(GraphLinkPredictor, self).__init__()
        logger.info(f"Initializing GraphLinkPredictor with in_channels={in_channels}, hidden={hidden_channels}, out={out_channels}, heads={nhead}, bert='{model_name}'")
        self.in_channels = in_channels
        self.mask_token = nn.Parameter(torch.zeros(in_channels if in_channels > 0 else 1))
        nn.init.xavier_uniform_(self.mask_token.data.unsqueeze(0))
        logger.debug(f"Initialized mask_token with shape: {self.mask_token.shape}")

        self.gat1 = GATConv(in_channels if in_channels > 0 else 1, hidden_channels, heads=nhead, dropout=0.3)
        self.ln1 = nn.LayerNorm(hidden_channels * nhead)
        self.dropout1 = nn.Dropout(0.1)
        self.gat2 = GATConv(hidden_channels * nhead, out_channels, heads=2, concat=True, dropout=0.3)
        self.ln2 = nn.LayerNorm(out_channels * 2)
        self.dropout2 = nn.Dropout(0.1)
        gnn_output_dim = out_channels * 2
        logger.debug(f"GNN architecture: GAT({self.gat1.in_channels}, {hidden_channels}*h{nhead}) -> LN -> ELU -> Dropout -> GAT({hidden_channels*nhead}, {out_channels}*h2) -> LN -> ELU -> Dropout. Output dim: {gnn_output_dim}")

        try:
            logger.info(f"Loading BERT tokenizer: {model_name}")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loading BERT model: {model_name}")
            self.bert_encoder = AutoModel.from_pretrained(model_name)
            self.bert_hidden_size = self.bert_encoder.config.hidden_size
            logger.info(f"BERT model loaded successfully. Hidden size: {self.bert_hidden_size}")
        except Exception as e:
            logger.warning(f"Failed to load BERT model '{model_name}'. BERT features will not be used. Error: {e}")
            self.bert_encoder = None
            self.bert_tokenizer = None
            self.bert_hidden_size = 0

        if self.bert_encoder is not None and self.bert_hidden_size > 0:
            self.proj_gnn = nn.Linear(gnn_output_dim, self.bert_hidden_size)
            fusion_hidden = 512
            self.fusion_network = nn.Sequential(
                nn.Linear(self.bert_hidden_size * 2, fusion_hidden),
                nn.LayerNorm(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_hidden, self.bert_hidden_size),
                nn.LayerNorm(self.bert_hidden_size)
            )
            final_node_embedding_dim = self.bert_hidden_size
            logger.debug(f"Initialized projection layer: Linear({gnn_output_dim}, {self.bert_hidden_size})")
            logger.debug(f"Initialized fusion network: Linear({self.bert_hidden_size * 2}, {fusion_hidden}) -> ... -> Linear({fusion_hidden}, {self.bert_hidden_size})")
        else:
            final_node_embedding_dim = gnn_output_dim
            self.proj_gnn = None
            self.fusion_network = None
            logger.debug("BERT or fusion components not initialized. Using GNN output directly.")

        predictor_hidden = final_node_embedding_dim
        self.link_predictor = nn.Sequential(
            nn.Linear(final_node_embedding_dim * 2, predictor_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(predictor_hidden, 1)
        )
        logger.debug(f"Initialized link predictor: Linear({final_node_embedding_dim * 2}, {predictor_hidden}) -> ReLU -> Dropout -> Linear({predictor_hidden}, 1)")

        self.bert_cache = None
        self.bert_texts_hash = None
        logger.info("GraphLinkPredictor initialization complete.")

    def ensure_mask_token_dim(self, feature_dim):
        if feature_dim <= 0:
             logger.warning("ensure_mask_token_dim: feature_dim is non-positive, skipping update.")
             return
        if self.mask_token.shape[0] != feature_dim:
            current_device = self.mask_token.device
            logger.info(f"ensure_mask_token_dim: Updating mask_token and GAT input dimension from {self.mask_token.shape[0]} to {feature_dim}.")
            self.mask_token = nn.Parameter(torch.zeros(feature_dim, device=current_device))
            nn.init.xavier_uniform_(self.mask_token.data.unsqueeze(0))
            if self.gat1.in_channels != feature_dim:
                hidden_channels = self.gat1.out_channels // self.gat1.heads
                nhead = self.gat1.heads
                dropout = self.gat1.dropout
                self.gat1 = GATConv(feature_dim, hidden_channels, heads=nhead, dropout=dropout).to(current_device)
                logger.debug(f"Reinitialized GATConv1 with in_channels={feature_dim}")
            self.in_channels = feature_dim

    def get_gnn_node_embeddings(self, x, edge_index):
        num_nodes = x.size(0)
        if num_nodes == 0:
            logger.debug("get_gnn_node_embeddings: num_nodes is 0, returning empty tensor.")
            gat2_out_dim = self.gat2.out_channels * self.gat2.heads
            return torch.zeros(0, gat2_out_dim, device=x.device)

        if edge_index is None or edge_index.numel() == 0:
             logger.warning("get_gnn_node_embeddings: edge_index is None or empty. Returning zero embeddings.")
             gat2_out_dim = self.gat2.out_channels * self.gat2.heads
             return torch.zeros(num_nodes, gat2_out_dim, device=x.device)
        if edge_index.max() >= num_nodes:
             logger.error(f"get_gnn_node_embeddings: edge_index max value ({edge_index.max()}) >= num_nodes ({num_nodes}). Returning zero embeddings.")
             gat2_out_dim = self.gat2.out_channels * self.gat2.heads
             return torch.zeros(num_nodes, gat2_out_dim, device=x.device)

        try:
            logger.debug(f"get_gnn_node_embeddings: Input x shape: {x.shape}, edge_index shape: {edge_index.shape}")
            x = self.gat1(x, edge_index)
            x = self.ln1(x)
            x = F.elu(x)
            x = self.dropout1(x)
            logger.debug(f"get_gnn_node_embeddings: After GAT1/LN/ELU/Dropout: {x.shape}")
            x = self.gat2(x, edge_index)
            x = self.ln2(x)
            x = F.elu(x)
            x = self.dropout2(x)
            logger.debug(f"get_gnn_node_embeddings: Final GNN embeddings shape: {x.shape}")
            return x
        except Exception as e:
            logger.error(f"get_gnn_node_embeddings: Error during GNN forward pass: {e}. Returning zero embeddings.", exc_info=True)
            gat2_out_dim = self.gat2.out_channels * self.gat2.heads
            return torch.zeros(num_nodes, gat2_out_dim, device=x.device)

    def get_bert_node_embeddings(self, num_nodes, device):
        if self.bert_encoder is None or self.bert_hidden_size == 0:
            logger.debug("get_bert_node_embeddings: BERT encoder not available. Returning zero tensor.")
            return torch.zeros(num_nodes, 1, device=device) # Return shape (N, 1) to avoid downstream errors if fusion expects specific dim

        texts = ["graph node"] * num_nodes
        current_hash = hash(tuple(texts))
        logger.debug(f"get_bert_node_embeddings: Requesting BERT embeddings for {num_nodes} nodes.")

        if self.bert_cache is not None:
            if (self.bert_texts_hash == current_hash and
                self.bert_cache.shape[0] >= num_nodes and
                self.bert_cache.device == device):
                logger.debug("get_bert_node_embeddings: Returning cached BERT embeddings.")
                return self.bert_cache[:num_nodes]
            else:
                logger.debug("get_bert_node_embeddings: Cache miss or invalid. Recomputing BERT embeddings.")

        self.bert_texts_hash = current_hash
        bert_embeddings = torch.zeros(num_nodes, self.bert_hidden_size, device=device)

        if not texts:
            logger.debug("get_bert_node_embeddings: No texts provided, returning zero embeddings.")
            return bert_embeddings

        try:
            bert_batch_size = 32
            self.bert_encoder.to(device)
            logger.debug(f"get_bert_node_embeddings: Starting BERT encoding with batch size {bert_batch_size}.")
            bert_pbar = tqdm(range(0, num_nodes, bert_batch_size), desc="BERT Encoding", leave=False, disable=num_nodes < bert_batch_size*2)
            for i in bert_pbar:
                batch_texts = texts[i : i + bert_batch_size]
                encodings = self.bert_tokenizer.batch_encode_plus(
                    batch_texts,
                    padding='longest',
                    truncation=True,
                    max_length=16,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad(), torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=torch.cuda.is_available()):
                    bert_output = self.bert_encoder(**encodings)

                batch_bert_embeddings = bert_output.last_hidden_state[:, 0, :]
                bert_embeddings[i : i + bert_batch_size] = batch_bert_embeddings

                del encodings, bert_output, batch_bert_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.debug(f"get_bert_node_embeddings: Finished BERT encoding. Output shape: {bert_embeddings.shape}")

        except Exception as e:
            logger.error(f"get_bert_node_embeddings: Error during BERT encoding: {e}. Returning zero embeddings.", exc_info=True)
            bert_embeddings = torch.zeros(num_nodes, self.bert_hidden_size, device=device)

        self.bert_cache = bert_embeddings
        return self.bert_cache

    def get_fused_node_embeddings(self, x, edge_index):
        logger.debug("get_fused_node_embeddings: Generating GNN embeddings.")
        gnn_embeds = self.get_gnn_node_embeddings(x, edge_index)
        logger.debug(f"get_fused_node_embeddings: GNN embeddings shape: {gnn_embeds.shape}")

        if self.bert_encoder is not None and self.fusion_network is not None and self.proj_gnn is not None:
            num_nodes = x.size(0)
            if num_nodes == 0:
                logger.debug("get_fused_node_embeddings: num_nodes is 0, returning empty tensor for fused embeddings.")
                return torch.zeros(0, self.bert_hidden_size, device=x.device)

            current_device = x.device
            logger.debug("get_fused_node_embeddings: Generating BERT embeddings.")
            bert_embeds = self.get_bert_node_embeddings(num_nodes, current_device)
            logger.debug(f"get_fused_node_embeddings: BERT embeddings shape: {bert_embeds.shape}")

            if gnn_embeds.size(0) != bert_embeds.size(0):
                 logger.warning(f"get_fused_node_embeddings: Mismatch in GNN ({gnn_embeds.size(0)}) and BERT ({bert_embeds.size(0)}) embedding counts. Returning GNN embeddings only.")
                 return gnn_embeds

            if gnn_embeds.size(1) != self.proj_gnn.in_features:
                 logger.warning(f"get_fused_node_embeddings: GNN embedding dimension ({gnn_embeds.size(1)}) mismatch with projection layer input ({self.proj_gnn.in_features}). Returning GNN embeddings only.")
                 return gnn_embeds

            logger.debug("get_fused_node_embeddings: Projecting GNN embeddings.")
            gnn_embeds_proj = self.proj_gnn(gnn_embeds)
            logger.debug(f"get_fused_node_embeddings: Projected GNN embeddings shape: {gnn_embeds_proj.shape}")

            expected_fusion_input_dim = self.fusion_network[0].in_features
            actual_fusion_input_dim = gnn_embeds_proj.size(1) + bert_embeds.size(1)
            if actual_fusion_input_dim != expected_fusion_input_dim:
                 logger.warning(f"get_fused_node_embeddings: Combined embedding dimension ({actual_fusion_input_dim}) mismatch with fusion network input ({expected_fusion_input_dim}). Returning GNN embeddings only.")
                 return gnn_embeds

            logger.debug("get_fused_node_embeddings: Concatenating projected GNN and BERT embeddings.")
            combined = torch.cat([gnn_embeds_proj, bert_embeds], dim=1)
            logger.debug(f"get_fused_node_embeddings: Combined embeddings shape: {combined.shape}")

            logger.debug("get_fused_node_embeddings: Applying fusion network.")
            fused_embeddings = self.fusion_network(combined)
            logger.debug(f"get_fused_node_embeddings: Fused embeddings shape: {fused_embeddings.shape}")
            return fused_embeddings
        else:
            logger.debug("get_fused_node_embeddings: BERT/Fusion components not available. Returning GNN embeddings.")
            return gnn_embeds

    def forward(self, x, edge_index, edge_label_index):
        logger.debug(f"forward: Input x shape: {x.shape if x is not None else 'None'}, edge_index shape: {edge_index.shape if edge_index is not None else 'None'}, edge_label_index shape: {edge_label_index.shape}")
        node_embeddings = self.get_fused_node_embeddings(x, edge_index)
        logger.debug(f"forward: Node embeddings shape: {node_embeddings.shape}")

        if node_embeddings.size(0) == 0:
             logger.warning("forward: Node embeddings are empty. Returning zero logits.")
             return torch.zeros(edge_label_index.size(1), device=x.device if x is not None else device)

        src_node_idx = edge_label_index[0]
        dst_node_idx = edge_label_index[1]

        max_node_idx = node_embeddings.size(0) - 1
        if src_node_idx.max() > max_node_idx or dst_node_idx.max() > max_node_idx:
            logger.error(f"forward: Edge index out of bounds. Max node index: {max_node_idx}, Max src: {src_node_idx.max()}, Max dst: {dst_node_idx.max()}. Returning zero logits.")
            return torch.zeros(edge_label_index.size(1), device=node_embeddings.device)
        if src_node_idx.min() < 0 or dst_node_idx.min() < 0:
            logger.error(f"forward: Negative node index found in edge_label_index. Min src: {src_node_idx.min()}, Min dst: {dst_node_idx.min()}. Returning zero logits.")
            return torch.zeros(edge_label_index.size(1), device=node_embeddings.device)

        logger.debug("forward: Gathering source and destination node embeddings.")
        src_embeds = node_embeddings[src_node_idx]
        dst_embeds = node_embeddings[dst_node_idx]
        logger.debug(f"forward: Source embeddings shape: {src_embeds.shape}, Dest embeddings shape: {dst_embeds.shape}")

        edge_features = torch.cat([src_embeds, dst_embeds], dim=1)
        logger.debug(f"forward: Concatenated edge features shape: {edge_features.shape}")

        expected_predictor_input_dim = self.link_predictor[0].in_features
        if edge_features.shape[1] != expected_predictor_input_dim:
            logger.error(f"forward: Edge feature dimension ({edge_features.shape[1]}) mismatch with predictor input ({expected_predictor_input_dim}). Returning zero logits.")
            return torch.zeros(edge_label_index.size(1), device=node_embeddings.device)

        logger.debug("forward: Applying link predictor.")
        link_logits = self.link_predictor(edge_features)
        logger.debug(f"forward: Link logits shape (before squeeze): {link_logits.shape}")

        return link_logits.squeeze(-1)

    def forward_pretrain(self, x, edge_index):
        logger.debug("forward_pretrain: Generating embeddings for pretraining.")
        return self.get_fused_node_embeddings(x, edge_index)

def pretrain_contrastive(model, data, pretrain_epochs=20, temp=0.5, mask_min=0.2, mask_max=0.4, beta=0.7, verbose=True, contrastive_batch_size=None, lr=1e-4, weight_decay=0.01):
    logger.info(f"Starting contrastive pretraining for {pretrain_epochs} epochs.")
    model.train()

    if not hasattr(data, 'x') or data.x is None:
        logger.warning("pretrain_contrastive: Data has no features (data.x is None). Skipping pretraining.")
        return 0.0
    logger.info(f"pretrain_contrastive: Ensuring mask token dimension matches data features ({data.x.shape[1]}).")
    model.ensure_mask_token_dim(data.x.shape[1])

    edge_index_for_pretrain = getattr(data, 'train_pos_edge_index', getattr(data, 'edge_index', None))
    if edge_index_for_pretrain is None:
        logger.warning("pretrain_contrastive: No suitable edge index found for pretraining. Skipping.")
        return 0.0
    logger.info(f"pretrain_contrastive: Using edge index with shape {edge_index_for_pretrain.shape} for pretraining.")

    params_to_optimize = list(model.gat1.parameters()) + list(model.ln1.parameters()) + \
                         list(model.gat2.parameters()) + list(model.ln2.parameters())
    if model.bert_encoder is not None and model.proj_gnn is not None and model.fusion_network is not None:
        params_to_optimize += list(model.proj_gnn.parameters())
        params_to_optimize += list(model.fusion_network.parameters())
        logger.debug("pretrain_contrastive: Added projection and fusion network parameters to optimizer.")
    params_to_optimize += [model.mask_token]
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    if not params_to_optimize:
        logger.warning("pretrain_contrastive: No trainable parameters found for pretraining. Skipping.")
        return 0.0
    logger.info(f"pretrain_contrastive: Optimizing {len(params_to_optimize)} parameter tensors.")

    optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs, eta_min=1e-6)
    logger.info(f"pretrain_contrastive: Optimizer: AdamW (lr={lr}, wd={weight_decay}), Scheduler: CosineAnnealingLR (T_max={pretrain_epochs})")

    total_loss_epoch = []
    start_time = time.time()
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    logger.info(f"pretrain_contrastive: GradScaler enabled: {torch.cuda.is_available()}")

    pretrain_pbar = tqdm(range(pretrain_epochs), desc="Pretraining", leave=False, disable=not verbose)
    for epoch in pretrain_pbar:
        model.train()
        optimizer.zero_grad()

        mask_ratio1 = random.uniform(mask_min, mask_max)
        mask_ratio2 = random.uniform(mask_min, mask_max)
        logger.debug(f"pretrain_contrastive Epoch {epoch+1}: Generating masks with ratios {mask_ratio1:.2f}, {mask_ratio2:.2f}")
        node_mask1 = generate_semantic_mask(data, mask_ratio1)
        node_mask2 = generate_semantic_mask(data, mask_ratio2)

        x_on_device = data.x.to(device)
        edge_index_on_device = edge_index_for_pretrain.to(device)

        logger.debug(f"pretrain_contrastive Epoch {epoch+1}: Applying soft masking.")
        x1 = soft_masking(x_on_device, node_mask1, model.mask_token, beta=beta)
        x2 = soft_masking(x_on_device, node_mask2, model.mask_token, beta=beta)

        try:
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=torch.cuda.is_available()):
                logger.debug(f"pretrain_contrastive Epoch {epoch+1}: Forward pass view 1.")
                z1 = model.forward_pretrain(x1, edge_index_on_device)
                logger.debug(f"pretrain_contrastive Epoch {epoch+1}: Forward pass view 2.")
                z2 = model.forward_pretrain(x2, edge_index_on_device)

                if z1.numel() == 0 or z2.numel() == 0 or z1.shape[0] != z2.shape[0]:
                    logger.warning(f"pretrain_contrastive Epoch {epoch+1}: Invalid embeddings generated (z1: {z1.shape}, z2: {z2.shape}). Skipping epoch.")
                    continue

                logger.debug(f"pretrain_contrastive Epoch {epoch+1}: Calculating NT-Xent loss (temp={temp}, batch_size={contrastive_batch_size}).")
                loss = nt_xent_loss(z1, z2, temperature=temp, batch_size=contrastive_batch_size)

            if torch.isnan(loss):
                logger.warning(f"pretrain_contrastive Epoch {epoch+1}: NaN loss encountered. Skipping backward pass.")
                continue

            logger.debug(f"pretrain_contrastive Epoch {epoch+1}: Backward pass.")
            scaler.scale(loss).backward()

            params_for_clipping = filter(lambda p: p.requires_grad and p.grad is not None, model.parameters())
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clipping, max_norm=1.0)
            logger.debug(f"pretrain_contrastive Epoch {epoch+1}: Gradient norm before clipping: {grad_norm:.4f}")

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            avg_epoch_loss = loss.item()
            total_loss_epoch.append(avg_epoch_loss)
            pretrain_pbar.set_postfix(loss=f"{avg_epoch_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            logger.debug(f"pretrain_contrastive Epoch {epoch+1}: Loss = {avg_epoch_loss:.4f}")

            if np.isnan(avg_epoch_loss):
                logger.error(f"pretrain_contrastive Epoch {epoch+1}: Loss became NaN. Stopping pretraining.")
                break

        except Exception as e:
            logger.error(f"pretrain_contrastive Epoch {epoch+1}: Error during pretraining loop: {e}. Stopping pretraining.", exc_info=True)
            break

    final_avg_loss = np.nanmean(total_loss_epoch[-5:]) if total_loss_epoch else 0.0
    elapsed_time = time.time() - start_time
    logger.info(f"Contrastive pretraining finished in {elapsed_time:.2f} seconds. Final avg loss (last 5 epochs): {final_avg_loss:.4f}")
    return final_avg_loss

def train_link_prediction(model, data, optimizer, criterion, batch_size):
    model.train()
    logger.debug("Starting link prediction training epoch.")

    pos_train_edge = data.train_pos_edge_index.to(device)
    edge_index = data.edge_index.to(device)
    x = data.x.to(device) if data.x is not None else None

    if x is None:
        logger.warning("train_link_prediction: Data has no features (data.x is None). Skipping training epoch.")
        return 0.0
    if edge_index is None or edge_index.numel() == 0:
         logger.warning("train_link_prediction: edge_index is None or empty. Skipping training epoch.")
         return 0.0
    if pos_train_edge is None or pos_train_edge.numel() == 0:
         logger.warning("train_link_prediction: Positive training edges are None or empty. Skipping training epoch.")
         return 0.0
    logger.debug(f"train_link_prediction: x shape: {x.shape}, edge_index shape: {edge_index.shape}, pos_train_edge shape: {pos_train_edge.shape}")

    total_loss = total_examples = 0
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    logger.debug(f"train_link_prediction: GradScaler enabled: {torch.cuda.is_available()}")

    num_pos_edges = pos_train_edge.size(1)
    train_loader = torch.utils.data.DataLoader(range(num_pos_edges), batch_size=batch_size, shuffle=True, pin_memory=True)
    logger.debug(f"train_link_prediction: Created DataLoader with {len(train_loader)} batches (batch size: {batch_size}).")

    batch_pbar = tqdm(train_loader, desc="Training Batches", leave=False, disable=len(train_loader) < 5)
    for perm in batch_pbar:
        optimizer.zero_grad()

        batch_pos_edge = pos_train_edge[:, perm]
        logger.debug(f"train_link_prediction Batch: Positive edge shape: {batch_pos_edge.shape}")

        try:
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=torch.cuda.is_available()):
                logger.debug("train_link_prediction Batch: Forward pass (positive edges).")
                pos_out = model(x, edge_index, batch_pos_edge)

                num_neg_samples = batch_pos_edge.size(1)
                logger.debug(f"train_link_prediction Batch: Sampling {num_neg_samples} negative edges.")
                batch_neg_edge = negative_sampling(
                    edge_index=edge_index, num_nodes=data.num_nodes,
                    num_neg_samples=num_neg_samples, method='sparse').to(device)
                logger.debug(f"train_link_prediction Batch: Negative edge shape: {batch_neg_edge.shape}")

                logger.debug("train_link_prediction Batch: Forward pass (negative edges).")
                neg_out = model(x, edge_index, batch_neg_edge)

                out = torch.cat([pos_out, neg_out], dim=0)
                pos_label = torch.ones(pos_out.size(0), device=device)
                neg_label = torch.zeros(neg_out.size(0), device=device)
                label = torch.cat([pos_label, neg_label], dim=0)
                logger.debug(f"train_link_prediction Batch: Output shape: {out.shape}, Label shape: {label.shape}")

                logger.debug("train_link_prediction Batch: Calculating loss.")
                loss = criterion(out, label)

            if torch.isnan(loss):
                 logger.warning("train_link_prediction Batch: NaN loss encountered. Skipping batch.")
                 continue

            logger.debug("train_link_prediction Batch: Backward pass.")
            scaler.scale(loss).backward()

            params_for_clipping = filter(lambda p: p.requires_grad and p.grad is not None, model.parameters())
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clipping, max_norm=1.0)
            logger.debug(f"train_link_prediction Batch: Gradient norm before clipping: {grad_norm:.4f}")

            scaler.step(optimizer)
            scaler.update()

            num_examples = out.size(0)
            batch_loss = loss.item()
            total_loss += batch_loss * num_examples
            total_examples += num_examples
            batch_pbar.set_postfix(loss=f"{batch_loss:.4f}")
            logger.debug(f"train_link_prediction Batch: Loss = {batch_loss:.4f}")

        except Exception as e:
            logger.error(f"train_link_prediction Batch: Error during training loop: {e}. Skipping batch.", exc_info=True)
            continue

    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    logger.debug(f"Finished link prediction training epoch. Average loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def evaluate_link_prediction(model, data, split_name='val', batch_size=65536):
    model.eval()
    logger.debug(f"Starting evaluation for split: {split_name}")

    pos_edge = data[f'{split_name}_pos_edge_index'].to(device)
    neg_edge = data[f'{split_name}_neg_edge_index'].to(device)

    edge_index = data.edge_index.to(device)
    x = data.x.to(device) if data.x is not None else None

    if x is None:
        logger.warning(f"evaluate_link_prediction ({split_name}): Data has no features (data.x is None). Returning zero metrics.")
        return {'AUC': 0.0, 'AP': 0.0}
    if edge_index is None:
         logger.warning(f"evaluate_link_prediction ({split_name}): edge_index is None. Returning zero metrics.")
         return {'AUC': 0.0, 'AP': 0.0}
    if pos_edge is None or neg_edge is None:
         logger.warning(f"evaluate_link_prediction ({split_name}): Positive or negative edges for split are None. Returning zero metrics.")
         return {'AUC': 0.0, 'AP': 0.0}
    logger.debug(f"evaluate_link_prediction ({split_name}): x shape: {x.shape}, edge_index shape: {edge_index.shape}, pos_edge shape: {pos_edge.shape}, neg_edge shape: {neg_edge.shape}")

    pos_preds = []
    pos_loader = torch.utils.data.DataLoader(range(pos_edge.size(1)), batch_size=batch_size, pin_memory=True)
    logger.debug(f"evaluate_link_prediction ({split_name}): Evaluating positive edges with {len(pos_loader)} batches.")
    pos_pbar = tqdm(pos_loader, desc=f"Eval Pos {split_name}", leave=False, disable=len(pos_loader) < 5)
    for perm in pos_pbar:
        batch_pos_edge = pos_edge[:, perm]
        try:
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=torch.cuda.is_available()):
                batch_pos_preds = model(x, edge_index, batch_pos_edge)
            pos_preds.append(batch_pos_preds.cpu())
            logger.debug(f"evaluate_link_prediction ({split_name}) Pos Batch: Output shape: {batch_pos_preds.shape}")
        except Exception as e:
            logger.error(f"evaluate_link_prediction ({split_name}) Pos Batch: Error during forward pass: {e}. Appending zeros.", exc_info=True)
            pos_preds.append(torch.zeros(batch_pos_edge.size(1)))

    pos_pred = torch.cat(pos_preds, dim=0)
    logger.debug(f"evaluate_link_prediction ({split_name}): Concatenated positive predictions shape: {pos_pred.shape}")

    neg_preds = []
    neg_loader = torch.utils.data.DataLoader(range(neg_edge.size(1)), batch_size=batch_size, pin_memory=True)
    logger.debug(f"evaluate_link_prediction ({split_name}): Evaluating negative edges with {len(neg_loader)} batches.")
    neg_pbar = tqdm(neg_loader, desc=f"Eval Neg {split_name}", leave=False, disable=len(neg_loader) < 5)
    for perm in neg_pbar:
        batch_neg_edge = neg_edge[:, perm]
        try:
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=torch.cuda.is_available()):
                batch_neg_preds = model(x, edge_index, batch_neg_edge)
            neg_preds.append(batch_neg_preds.cpu())
            logger.debug(f"evaluate_link_prediction ({split_name}) Neg Batch: Output shape: {batch_neg_preds.shape}")
        except Exception as e:
            logger.error(f"evaluate_link_prediction ({split_name}) Neg Batch: Error during forward pass: {e}. Appending zeros.", exc_info=True)
            neg_preds.append(torch.zeros(batch_neg_edge.size(1)))

    neg_pred = torch.cat(neg_preds, dim=0)
    logger.debug(f"evaluate_link_prediction ({split_name}): Concatenated negative predictions shape: {neg_pred.shape}")

    predictions = torch.cat([pos_pred, neg_pred], dim=0)
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0)
    logger.debug(f"evaluate_link_prediction ({split_name}): Final predictions shape: {predictions.shape}, Labels shape: {labels.shape}")

    try:
        y_true = labels.numpy()
        y_scores = predictions.numpy()

        if len(np.unique(y_true)) < 2:
             logger.warning(f"evaluate_link_prediction ({split_name}): Only one class present in labels. Cannot compute AUC/AP.")
             auc_score = 0.0
             ap_score = 0.0
        else:
             auc_score = roc_auc_score(y_true, y_scores)
             ap_score = average_precision_score(y_true, y_scores)
             logger.debug(f"evaluate_link_prediction ({split_name}): AUC = {auc_score:.4f}, AP = {ap_score:.4f}")

    except Exception as e:
        logger.error(f"evaluate_link_prediction ({split_name}): Error calculating metrics: {e}. Returning zero metrics.", exc_info=True)
        auc_score = 0.0
        ap_score = 0.0

    results = {'AUC': auc_score, 'AP': ap_score}
    logger.info(f"Evaluation results for {split_name}: AUC={results['AUC']:.4f}, AP={results['AP']:.4f}")
    return results

def main():
    logger.info("Starting main execution.")
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    logger.info("Created results and checkpoints directories if they didn't exist.")

    num_final_runs = 7
    beta = 0.7
    mask_min = 0.2
    mask_max = 0.4
    pretrain_epochs = 30
    train_epochs = 120
    patience = 30
    eval_steps = 1
    batch_size = 1024
    eval_batch_size = 2048
    contrastive_batch_size = 512
    temp = 0.5
    hidden_channels = 128
    out_channels = 64
    nhead = 4
    bert_model_name = 'distilbert/distilbert-base-uncased'
    lr_graph = 1e-3
    lr_bert = 1e-5
    lr_other = 1e-4
    lr_pretrain = 1e-4
    weight_decay = 0.0

    logger.info("Hyperparameters set:")
    logger.info(f"  num_final_runs={num_final_runs}, beta={beta}, mask_range=[{mask_min},{mask_max}]")
    logger.info(f"  pretrain_epochs={pretrain_epochs}, train_epochs={train_epochs}, patience={patience}")
    logger.info(f"  eval_steps={eval_steps}, batch_size={batch_size}, eval_batch_size={eval_batch_size}")
    logger.info(f"  contrastive_batch_size={contrastive_batch_size}, temp={temp}")
    logger.info(f"  hidden_channels={hidden_channels}, out_channels={out_channels}, nhead={nhead}")
    logger.info(f"  bert_model_name='{bert_model_name}'")
    logger.info(f"  lr_graph={lr_graph}, lr_bert={lr_bert}, lr_other={lr_other}, lr_pretrain={lr_pretrain}, weight_decay={weight_decay}")

    dataset_configs = ['CiteSeer']
    all_datasets_results = {}

    if torch.__version__ >= '2.0.0':
        try:
            torch.set_float32_matmul_precision('high')
            logger.info("Set float32 matmul precision to 'high' (PyTorch >= 2.0.0).")
        except Exception as e:
            logger.warning(f"Could not set float32 matmul precision: {e}")

    dataset_pbar = tqdm(dataset_configs, desc="Datasets")
    for dataset_name in dataset_pbar:
        dataset_pbar.set_postfix(current=dataset_name)
        logger.info(f"Processing dataset: {dataset_name}")
        try:
            path = os.path.join('./data/Planetoid', dataset_name)
            logger.info(f"Loading dataset {dataset_name} from {path}")
            dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            logger.info(f"Dataset {dataset_name} loaded. Nodes: {data.num_nodes}, Features: {data.num_features}, Edges: {data.num_edges}")

            logger.info(f"Splitting edges for {dataset_name} (val=0.05, test=0.1)")
            data.train_mask = data.val_mask = data.test_mask = None
            data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)
            logger.info(f"Edge split complete for {dataset_name}:")
            logger.info(f"  Train pos edges: {data.train_pos_edge_index.shape[1] if data.train_pos_edge_index is not None else 'N/A'}")
            logger.info(f"  Val pos edges: {data.val_pos_edge_index.shape[1] if data.val_pos_edge_index is not None else 'N/A'}, Val neg edges: {data.val_neg_edge_index.shape[1] if data.val_neg_edge_index is not None else 'N/A'}")
            logger.info(f"  Test pos edges: {data.test_pos_edge_index.shape[1] if data.test_pos_edge_index is not None else 'N/A'}, Test neg edges: {data.test_neg_edge_index.shape[1] if data.test_neg_edge_index is not None else 'N/A'}")


            if data.train_pos_edge_index is not None:
                data.edge_index = data.train_pos_edge_index.clone()
                logger.info(f"Set data.edge_index to train_pos_edge_index for {dataset_name}.")
            else:
                logger.error(f"Missing train_pos_edge_index for {dataset_name} after split. Skipping dataset.")
                raise ValueError(f"Missing train_pos_edge_index for {dataset_name}")

            in_channels = data.num_features
            num_nodes = data.num_nodes
            logger.info(f"Dataset {dataset_name}: in_channels={in_channels}, num_nodes={num_nodes}")

            test_auc_runs = []
            test_ap_runs = []

            run_pbar = tqdm(range(num_final_runs), desc=f"{dataset_name} Runs", leave=False)
            for run in run_pbar:
                run_pbar.set_postfix(run=f"{run+1}/{num_final_runs}")
                logger.info(f"Starting run {run+1}/{num_final_runs} for dataset {dataset_name}")
                seed = run
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                logger.info(f"Set random seed to {seed} for run {run+1}")

                model = GraphLinkPredictor(in_channels=in_channels,
                                        hidden_channels=hidden_channels,
                                        out_channels=out_channels,
                                        nhead=nhead,
                                        model_name=bert_model_name).to(device)
                if torch.__version__ >= '2.0.0':
                    try:
                        model = torch.compile(model)
                        logger.info(f"Compiled model using torch.compile() for run {run+1}.")
                    except Exception as e:
                        logger.warning(f"torch.compile() failed for run {run+1}: {e}")


                if data.x is not None:
                    logger.info(f"Run {run+1}: Ensuring model mask token dim matches data features ({data.x.shape[1]}).")
                    model.ensure_mask_token_dim(data.x.shape[1])
                else:
                    logger.warning(f"Run {run+1}: data.x is None for {dataset_name}. Model input features might be incorrect if defaults are used.")


                pre_trained_state_dict = None
                if pretrain_epochs > 0:
                    logger.info(f"Run {run+1}: Starting pretraining ({pretrain_epochs} epochs).")
                    pretrain_loss = pretrain_contrastive(model, data, pretrain_epochs=pretrain_epochs,
                                        temp=temp, mask_min=mask_min, mask_max=mask_max, beta=beta,
                                        verbose=False, contrastive_batch_size=contrastive_batch_size,
                                        lr=lr_pretrain, weight_decay=weight_decay)
                    logger.info(f"Run {run+1}: Pretraining finished. Final avg loss: {pretrain_loss:.4f}")
                    pre_trained_state_dict = copy.deepcopy(model.state_dict())
                    logger.info(f"Run {run+1}: Saved pretrained model state dict.")

                    logger.info(f"Run {run+1}: Re-initializing model before fine-tuning.")
                    model = GraphLinkPredictor(in_channels=in_channels,
                                            hidden_channels=hidden_channels,
                                            out_channels=out_channels, nhead=nhead,
                                            model_name=bert_model_name).to(device)
                    if data.x is not None:
                         model.ensure_mask_token_dim(data.x.shape[1])

                    if pre_trained_state_dict:
                        try:
                            missing_keys, unexpected_keys = model.load_state_dict(pre_trained_state_dict, strict=False)
                            logger.info(f"Run {run+1}: Loaded pretrained state dict. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                            if missing_keys or unexpected_keys:
                                logger.debug(f"  Missing: {missing_keys}")
                                logger.debug(f"  Unexpected: {unexpected_keys}")
                        except Exception as e:
                            logger.error(f"Run {run+1}: Failed to load pretrained state dict: {e}", exc_info=True)
                            pass
                else:
                    logger.info(f"Run {run+1}: Skipping pretraining as pretrain_epochs=0.")


                graph_params = []
                bert_params = []
                other_params = []
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if model.bert_encoder is not None and name.startswith('bert_encoder.'):
                        bert_params.append(param)
                    elif name.startswith('gat') or name.startswith('ln') or \
                        (model.proj_gnn is not None and name.startswith('proj_gnn.')):
                        graph_params.append(param)
                    else:
                        other_params.append(param)
                logger.info(f"Run {run+1}: Parameter groups for optimizer: Graph={len(graph_params)}, BERT={len(bert_params)}, Other={len(other_params)}")

                optimizer_grouped_parameters = []
                if graph_params:
                    optimizer_grouped_parameters.append({'params': graph_params, 'lr': lr_graph, 'weight_decay': weight_decay})
                if bert_params:
                    optimizer_grouped_parameters.append({'params': bert_params, 'lr': lr_bert, 'weight_decay': weight_decay})
                if other_params:
                    optimizer_grouped_parameters.append({'params': other_params, 'lr': lr_other, 'weight_decay': weight_decay})

                if not optimizer_grouped_parameters:
                    logger.error(f"Run {run+1}: No parameters found for optimizer. Skipping run.")
                    continue

                optimizer = optim.AdamW(optimizer_grouped_parameters)
                criterion = torch.nn.BCEWithLogitsLoss()
                logger.info(f"Run {run+1}: Initialized AdamW optimizer and BCEWithLogitsLoss criterion.")

                best_val_metric = 0.0
                best_epoch = -1
                epochs_no_improve = 0
                best_model_state_dict = None

                logger.info(f"Run {run+1}: Starting fine-tuning for max {train_epochs} epochs (patience={patience}).")
                epoch_pbar = tqdm(range(1, train_epochs + 1), desc=f"Run {run+1} Epochs", leave=False)
                for epoch in epoch_pbar:
                    loss = train_link_prediction(model, data, optimizer, criterion, batch_size)
                    logger.debug(f"Run {run+1} Epoch {epoch}: Train Loss = {loss:.4f}")

                    if epoch % eval_steps == 0:
                        logger.debug(f"Run {run+1} Epoch {epoch}: Evaluating on validation set.")
                        val_results = evaluate_link_prediction(model, data, split_name='val', batch_size=eval_batch_size)
                        val_auc = val_results.get('AUC', 0.0)
                        val_ap = val_results.get('AP', 0.0)
                        logger.info(f"Run {run+1} Epoch {epoch}: Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}")

                        current_metric = val_auc # Using AUC for early stopping

                        if current_metric > best_val_metric:
                            best_val_metric = current_metric
                            best_epoch = epoch
                            epochs_no_improve = 0
                            best_model_state_dict = copy.deepcopy(model.state_dict())
                            logger.info(f"Run {run+1} Epoch {epoch}: New best validation AUC: {best_val_metric:.4f}. Saved model state.")
                        else:
                            epochs_no_improve += eval_steps
                            logger.debug(f"Run {run+1} Epoch {epoch}: No improvement. Epochs without improvement: {epochs_no_improve}/{patience}")

                        epoch_pbar.set_postfix(loss=f"{loss:.4f}", val_auc=f"{val_auc:.4f}", best_auc=f"{best_val_metric:.4f}")

                        if epochs_no_improve >= patience:
                            logger.info(f"Run {run+1}: Early stopping triggered at epoch {epoch} after {epochs_no_improve} epochs without improvement.")
                            break
                    else:
                         epoch_pbar.set_postfix(loss=f"{loss:.4f}", best_auc=f"{best_val_metric:.4f}")


                if best_model_state_dict is not None:
                    try:
                        logger.info(f"Run {run+1}: Loading best model state from epoch {best_epoch} (Val AUC: {best_val_metric:.4f}).")
                        model.load_state_dict(best_model_state_dict)
                    except Exception as e:
                        logger.error(f"Run {run+1}: Failed to load best model state dict: {e}", exc_info=True)
                        pass
                else:
                     logger.warning(f"Run {run+1}: No best model state was saved (possibly due to no improvement or error). Using last model state.")


                logger.info(f"Run {run+1}: Evaluating final model on test set.")
                test_results = evaluate_link_prediction(model, data, split_name='test', batch_size=eval_batch_size)
                test_auc = test_results.get('AUC', 0.0)
                test_ap = test_results.get('AP', 0.0)
                logger.info(f"Run {run+1} Test Results: AUC={test_auc:.4f}, AP={test_ap:.4f}")

                test_auc_runs.append(test_auc)
                test_ap_runs.append(test_ap)

                del model, optimizer, best_model_state_dict
                if 'pre_trained_state_dict' in locals():
                    del pre_trained_state_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Run {run+1} completed and cleaned up.")


            if test_auc_runs and test_ap_runs:
                mean_auc = np.mean(test_auc_runs)
                std_auc = np.std(test_auc_runs)
                mean_ap = np.mean(test_ap_runs)
                std_ap = np.std(test_ap_runs)
                all_datasets_results[dataset_name] = {
                    'mean_auc': mean_auc, 'std_auc': std_auc,
                    'mean_ap': mean_ap, 'std_ap': std_ap,
                    'runs': len(test_auc_runs)
                }
                logger.info(f"Dataset {dataset_name} finished. Avg Test AUC: {mean_auc:.4f} +/- {std_auc:.4f}, Avg Test AP: {mean_ap:.4f} +/- {std_ap:.4f} ({len(test_auc_runs)} runs)")
            else:
                all_datasets_results[dataset_name] = None
                logger.warning(f"Dataset {dataset_name} finished with no successful runs.")


            del data, dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Cleaned up data and dataset objects for {dataset_name}.")

        except Exception as e:
            logger.error(f"Failed to process dataset {dataset_name}: {e}", exc_info=True)
            all_datasets_results[dataset_name] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    print("\n\n" + "="*60)
    print("========= Final Summary Across All Datasets =========")
    print("="*60)
    logger.info("="*60)
    logger.info("========= Final Summary Across All Datasets =========")
    logger.info("="*60)
    for name, results in all_datasets_results.items():
        print(f"\n--- Dataset: {name} ---")
        logger.info(f"\n--- Dataset: {name} ---")
        if results:
            print(f"  (Based on {results['runs']}/{num_final_runs} successful runs)")
            print(f"  Average Test AUC: {results['mean_auc']:.4f} +/- {results['std_auc']:.4f}")
            print(f"  Average Test AP : {results['mean_ap']:.4f} +/- {results['std_ap']:.4f}")
            combined_avg = (results['mean_auc'] + results['mean_ap']) / 2
            print(f"  Combined Avg (AUC+AP)/2: {combined_avg:.4f}")
            logger.info(f"  (Based on {results['runs']}/{num_final_runs} successful runs)")
            logger.info(f"  Average Test AUC: {results['mean_auc']:.4f} +/- {results['std_auc']:.4f}")
            logger.info(f"  Average Test AP : {results['mean_ap']:.4f} +/- {results['std_ap']:.4f}")
            logger.info(f"  Combined Avg (AUC+AP)/2: {combined_avg:.4f}")
        else:
            print("  No successful runs completed or dataset failed to load.")
            logger.warning("  No successful runs completed or dataset failed to load.")
    print("\n" + "="*60)
    logger.info("\n" + "="*60)
    logger.info("Main execution finished.")

if __name__ == "__main__":
    base_seed = 42
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)
        torch.backends.cudnn.benchmark = True
        logger.info(f"Set base random seed to {base_seed} and enabled CUDNN benchmark.")
    else:
        logger.info(f"Set base random seed to {base_seed}.")

    main()