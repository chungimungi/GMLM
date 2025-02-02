import torch
import torch.optim as optim
from utils import generate_semantic_mask, soft_masking, nt_xent_loss
import random

def pretrain_contrastive(model, data, pretrain_epochs=20, temp=0.5):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    
    best_loss = float('inf')

    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        
        mask1 = generate_semantic_mask(data, random.uniform(0.2, 0.4))
        mask2 = generate_semantic_mask(data, random.uniform(0.2, 0.4))
        
        x1 = soft_masking(data.x, mask1, model.mask_token, beta=0.7)
        x2 = soft_masking(data.x, mask2, model.mask_token, beta=0.7)
        
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
        mask = generate_semantic_mask(data, random.uniform(0.2, 0.4))
        x_soft = soft_masking(data.x, mask, model.mask_token, beta=0.7)
        
        logits = model(x_soft, data.edge_index, mask)
        loss = criterion(logits[mask], data.y[mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"[Train] Epoch: {epoch}, Loss: {loss.item():.4f}")
