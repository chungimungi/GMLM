import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AutoModel, AutoTokenizer

class GraphMaskedLM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, nhead=4, model_name='distilbert/distilbert-base-uncased'):
        super(GraphMaskedLM, self).__init__()
        self.mask_token = nn.Parameter(torch.zeros(in_channels))
        
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