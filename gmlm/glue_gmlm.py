import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
import torchinfo
from tqdm import tqdm
import random
import warnings
import os
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
warnings.filterwarnings("ignore")

# ----------------- Models -----------------
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

        try:
            self.encoder = AutoModel.from_pretrained(model_name,
                                                    torchscript=True,
                                                    low_cpu_mem_usage=True)
            self.encoder.gradient_checkpointing_enable()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Failed to load model with optimized settings: {e}")
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

    def forward(self, x, edge_index, mask, text_inputs=None):
        x_graph = self.get_graph_embeddings(x, edge_index)

        # Reduce batch size to save memory
        batch_size = 64
        all_bert_embeddings = []
        mask_list = mask.cpu().tolist()

        # If text_inputs provided, use them instead of mask tokens
        if text_inputs is not None:
            encodings = self.tokenizer.batch_encode_plus(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(x.device)

            with torch.no_grad() if not self.training else torch.enable_grad():
                bert_output = self.encoder(**encodings)
                if isinstance(bert_output, tuple):
                    bert_embeddings = bert_output[0][:, 0, :]
                else:
                    bert_embeddings = bert_output.last_hidden_state[:, 0, :]
                
                if not self.training:
                    torch.cuda.empty_cache()
                    gc.collect()
        else:
            for i in tqdm(range(0, len(mask_list), batch_size), desc="Processing node batches", leave=False):
                batch_mask = mask_list[i:i+batch_size]
                texts = ["[MASK]" if flag else "node" for flag in batch_mask]

                encodings = self.tokenizer.batch_encode_plus(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors="pt"
                ).to(x.device)

                with torch.no_grad() if not self.training else torch.enable_grad():
                    bert_output = self.encoder(**encodings)
                    if isinstance(bert_output, tuple):
                        batch_embeddings = bert_output[0][:, 0, :]
                    else:
                        batch_embeddings = bert_output.last_hidden_state[:, 0, :]
                    all_bert_embeddings.append(batch_embeddings)

                    if not self.training:
                        torch.cuda.empty_cache()
                        gc.collect()

            bert_embeddings = torch.cat(all_bert_embeddings, dim=0)

        if text_inputs is not None:
            node_indices = torch.arange(len(text_inputs), device=x.device)
            x_graph_selected = x_graph[node_indices]
            combined = torch.cat([x_graph_selected, bert_embeddings], dim=1)
        else:
            combined = torch.cat([x_graph, bert_embeddings], dim=1)
            
        fused = self.fusion_network(combined)
        logits = self.classifier(fused)
        return logits

# ----------------- Utility Functions -----------------
def generate_semantic_mask(data, mask_ratio):
    if hasattr(data, 'train_mask'):
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        num_train = train_idx.size(0)
        num_mask = max(1, int(mask_ratio * num_train))

        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        train_degrees = deg[train_idx]

        probs = train_degrees / train_degrees.sum()
        sampled = train_idx[torch.multinomial(probs, num_mask, replacement=False)]
    else:
        train_nodes = torch.arange(data.num_nodes, device=data.x.device)
        num_train = train_nodes.size(0)
        num_mask = max(1, int(mask_ratio * num_train))

        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        train_degrees = deg[train_nodes]

        probs = train_degrees / train_degrees.sum()
        sampled = train_nodes[torch.multinomial(probs, num_mask, replacement=False)]

    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    mask[sampled] = True
    return mask

def soft_masking(x, mask, mask_token, beta=0.7):
    device = x.device
    mask = mask.to(device)
    mask_token = mask_token.to(device)

    x_masked = x.clone()
    x_masked[mask] = (1 - beta) * x[mask] + beta * mask_token
    return x_masked

def nt_xent_loss(z1, z2, temperature=0.5, batch_size=64):
    device = z1.device
    total_loss = 0.0
    num_batches = 0
    total_samples = z1.size(0)

    for i in tqdm(range(0, total_samples, batch_size), desc="Computing contrastive loss", leave=False):
        end_idx = min(i + batch_size, total_samples)
        batch_size_curr = end_idx - i

        if batch_size_curr <= 1:
            continue

        batch_z1 = F.normalize(z1[i:end_idx], dim=1)
        batch_z2 = F.normalize(z2[i:end_idx], dim=1)

        batch_emb = torch.cat([batch_z1, batch_z2], dim=0)

        similarity = torch.mm(batch_emb, batch_emb.t()) / temperature

        mask_mat = torch.eye(2 * batch_size_curr, dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(mask_mat, -9e15)

        pos_indices = (torch.arange(2 * batch_size_curr, device=device) + batch_size_curr) % (2 * batch_size_curr)

        batch_loss = F.cross_entropy(similarity, pos_indices)

        total_loss += batch_loss
        num_batches += 1

        torch.cuda.empty_cache()
        gc.collect()

    if num_batches == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / num_batches

# ----------------- Load Local GLUE Datasets -----------------
def load_local_glue_dataset(task_name, glue_dir):
    print(f"Loading local {task_name} dataset...")
    task_map = {
        'cola': 'CoLA',
        'sst2': 'SST-2',
        'sts-b': 'STS-B',
        'mrpc': 'MRPC',
        'qqp': 'QQP',
        'mnli': 'MNLI',
        'qnli': 'QNLI',
        'rte': 'RTE',
        'wnli': 'WNLI'
    }
    
    folder_name = task_map.get(task_name.lower(), task_name)
    task_path = os.path.join(glue_dir, folder_name)
    
    if not os.path.exists(task_path):
        raise ValueError(f"Task directory {task_path} not found")
    
    result = {}
    
    train_path = os.path.join(task_path, "train.tsv")
    if os.path.exists(train_path):
        try:
            train_df = pd.read_csv(train_path, sep='\t', quoting=3)
            result['train'] = train_df
            print(f"Loaded training data with {len(train_df)} examples")
        except Exception as e:
            print(f"Error loading train data: {e}")
    
    test_path = os.path.join(task_path, "dev.tsv")
    if os.path.exists(test_path):
        try:
            test_df = pd.read_csv(test_path, sep='\t', quoting=3)
            result['test'] = test_df
            print(f"Loaded test data with {len(test_df)} examples")
        except Exception as e:
            print(f"Error loading test data: {e}")
    
    return result

def extract_text_and_labels(df, task_name):
    if task_name.lower() == 'cola':
        if len(df.columns) >= 4:  
            texts = df.iloc[:, 3].tolist()
            labels = df.iloc[:, 1].tolist()
        elif 'sentence' in df.columns and 'label' in df.columns:
            texts = df['sentence'].tolist()
            labels = df['label'].tolist()
        else:
            if df.columns[0] == 'index':
                texts = df.iloc[:, 3].tolist() if len(df.columns) > 3 else df.iloc[:, 1].tolist()
                labels = df.iloc[:, 1].tolist() if len(df.columns) > 3 else df.iloc[:, 0].tolist()
            else:
                texts = df.iloc[:, -1].tolist()  
                labels = df.iloc[:, 1].tolist()  
    
    elif task_name.lower() == 'sts-b':
        # STS-B is now a separate case, not inside the CoLA condition
        if 'sentence1' in df.columns and 'sentence2' in df.columns:
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(df['sentence1'], df['sentence2'])]
        else:
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(df.iloc[:, 0], df.iloc[:, 1])]
        
        if 'score' in df.columns:
            labels = df['score'].tolist()
        else:
            labels = df.iloc[:, -1].tolist()
    elif task_name.lower() == 'sst2':
        if 'sentence' in df.columns and 'label' in df.columns:
            texts = df['sentence'].tolist()
            labels = df['label'].tolist()
        else:
            texts = df.iloc[:, 0].tolist()
            labels = df.iloc[:, 1].tolist() if len(df.columns) > 1 else [0] * len(df)
            
    elif task_name.lower() in ['mrpc', 'qqp']:
        if 'sentence1' in df.columns and 'sentence2' in df.columns:
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(df['sentence1'], df['sentence2'])]
        elif '#1 String' in df.columns and '#2 String' in df.columns:
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(df['#1 String'], df['#2 String'])]
        elif len(df.columns) >= 5:
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(df.iloc[:, 3], df.iloc[:, 4])]
        else:
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(df.iloc[:, 0], df.iloc[:, 1])]
            
        if 'label' in df.columns:
            labels = df['label'].tolist()
        elif 'Quality' in df.columns:
            labels = df['Quality'].tolist()
        elif len(df.columns) > 0 and df.columns[0] == 'index':
            labels = df.iloc[:, 1].tolist()
        else:
            labels = df.iloc[:, -1].tolist() if len(df.columns) > 2 else [0] * len(df)
            
    elif task_name.lower() in ['mnli', 'qnli', 'rte', 'wnli']:
        if 'sentence1' in df.columns and 'sentence2' in df.columns:
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(df['sentence1'], df['sentence2'])]
        elif 'premise' in df.columns and 'hypothesis' in df.columns:
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(df['premise'], df['hypothesis'])]
        elif 'question' in df.columns and 'sentence' in df.columns:
            texts = [f"{q} [SEP] {s}" for q, s in zip(df['question'], df['sentence'])]
        else:
            col_names = df.columns.tolist()
            text_col1 = next((col for col in col_names if 'sent' in col.lower() or 'prem' in col.lower() 
                             or 'quest' in col.lower()), col_names[0])
            text_col2 = next((col for col in col_names if ('sent' in col.lower() and col != text_col1)
                             or 'hypo' in col.lower() or 'answ' in col.lower()), col_names[1])
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(df[text_col1], df[text_col2])]
        
        if 'label' in df.columns:
            labels = df['label'].tolist()
        elif 'gold_label' in df.columns:
            labels = df['gold_label'].tolist()
        else:
            label_col = next((col for col in df.columns if 'label' in col.lower()), None)
            if label_col:
                labels = df[label_col].tolist()
            else:
                labels = [0] * len(df)
    
    else:
        if len(df.columns) >= 2:
            texts = df.iloc[:, 0].tolist()
            labels = df.iloc[:, 1].tolist()
        else:
            texts = df.iloc[:, 0].tolist()
            labels = [0] * len(df)
    
    return texts, labels

def create_graph_data_from_local_dataset(df, task_name, max_nodes=10000):
    texts, labels = extract_text_and_labels(df, task_name)
    
    if len(texts) > max_nodes:
        indices = np.random.choice(len(texts), max_nodes, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    num_nodes = len(texts)
    node_features = torch.randn(num_nodes, 128)
    
    edge_index = torch.zeros((2, num_nodes-1), dtype=torch.long)
    edge_index[0] = torch.arange(0, num_nodes-1)
    edge_index[1] = torch.arange(1, num_nodes)
    
    data = Data(x=node_features, edge_index=edge_index)
    data.num_nodes = num_nodes
    data.train_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    return data, texts, labels

# ----------------- Training Functions -----------------
def pretrain_contrastive(model, data, pretrain_epochs=20, temp=0.5):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)

    pbar = tqdm(range(pretrain_epochs), desc="Pretraining")
    for epoch in pbar:
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

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        torch.cuda.empty_cache()
        gc.collect()

def train_glue(model, data, texts, labels, task_name, num_epochs=5):
    model.train()
    
    # STS-B is a regression task
    is_regression = task_name.lower() == "sts-b"
    is_binary = task_name.lower() in ["cola", "sst2", "mrpc", "qqp"]
    
    if is_regression:
        criterion = torch.nn.MSELoss()
    elif is_binary:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    device = data.x.device
    if isinstance(labels[0], str):
        unique_labels = sorted(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels = [label_map[label] for label in labels]
    
    # For regression tasks, convert to float
    if is_regression:
        labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    else:
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    pbar = tqdm(range(num_epochs), desc=f"Training on {task_name}")
    num_samples = len(texts)
    for epoch in pbar:
        batch_size = 64
        running_loss = 0.0
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_texts = texts[i:end_idx]
            batch_labels = labels_tensor[i:end_idx]
            
            optimizer.zero_grad()
            mask = generate_semantic_mask(data, random.uniform(0.1, 0.3))
            x_soft = soft_masking(data.x, mask, model.mask_token, beta=0.5)
            
            with torch.amp.autocast('cuda'):
                logits = model(x_soft, data.edge_index, mask, text_inputs=batch_texts)
                
                if is_regression:
                    # For regression tasks, ensure the output is squeezed to match labels
                    logits = logits.squeeze(-1)
                    loss = criterion(logits, batch_labels)
                elif is_binary and logits.dim() > 1 and logits.size(1) == 2:
                    loss = criterion(logits, batch_labels)
                else:
                    loss = criterion(logits, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            torch.cuda.empty_cache()
            gc.collect()
        
        scheduler.step()
        pbar.set_postfix({"loss": f"{running_loss / (num_samples // batch_size + 1):.4f}"})

def evaluate_glue(model, data, texts, labels, task_name):
    model.eval()
    device = data.x.device
    
    # Handle STS-B as regression task
    is_regression = task_name.lower() == "sts-b"
    
    if isinstance(labels[0], str) and not is_regression:
        unique_labels = sorted(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels = [label_map[label] for label in labels]
    
    if is_regression:
        # Convert regression labels to float
        labels = [float(label) for label in labels]
    
    batch_size = 64
    num_samples = len(texts)
    all_preds = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc=f"Evaluating {task_name}"):
            end_idx = min(i + batch_size, num_samples)
            batch_texts = texts[i:end_idx]
            mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            logits = model(data.x, data.edge_index, mask, text_inputs=batch_texts)
            
            if is_regression:
                # For regression, take raw predictions
                preds = logits.squeeze(-1).cpu().numpy()
            elif task_name.lower() in ["cola", "sst2", "mrpc", "qqp"]:
                if logits.dim() > 1 and logits.size(1) == 2:
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                else:
                    preds = (torch.sigmoid(logits.view(-1)) > 0.5).cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
            all_preds.extend(preds.tolist())
    
    results = {}
    if is_regression:
        from scipy.stats import pearsonr, spearmanr
        pearson_corr, _ = pearsonr(labels, all_preds)
        spearman_corr, _ = spearmanr(labels, all_preds)
        print(f"Pearson correlation: {pearson_corr:.4f}")
        print(f"Spearman correlation: {spearman_corr:.4f}")
        results["pearson"] = pearson_corr
        results["spearman"] = spearman_corr
        results["avg_correlation"] = (pearson_corr + spearman_corr) / 2
    elif task_name.lower() == "cola":
        mcc = matthews_corrcoef(labels, all_preds)
        print(f"Matthews Correlation: {mcc:.4f}")
        results["matthews_correlation"] = mcc
    else:
        accuracy = accuracy_score(labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")
        results["accuracy"] = accuracy
        
        if task_name.lower() in ["mrpc", "qqp"]:
            f1 = f1_score(labels, all_preds)
            print(f"F1 Score: {f1:.4f}")
            results["f1"] = f1
    
    return results

# ----------------- Main Execution with Grid Search -----------------
if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    glue_dir = "glue_data" 
    task_name = "sts-b"  
    print(f"Loading GLUE task: {task_name}...")
    
    try:
        glue_dataset = load_local_glue_dataset(task_name, glue_dir)
        print(f"Successfully loaded {task_name} dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    if 'train' in glue_dataset:
        train_data, train_texts, train_labels = create_graph_data_from_local_dataset(
            glue_dataset['train'], task_name, max_nodes=10000
        )
    else:
        print("Warning: No training data found!")
        exit(1)
    
    if 'test' in glue_dataset:
        test_data, test_texts, test_labels = create_graph_data_from_local_dataset(
            glue_dataset['test'], task_name, max_nodes=1000
        )
    else:
        print("Warning: No test data found!")
        test_data, test_texts, test_labels = None, None, None
    
    train_data = train_data.to(device)
    
    # In the main execution section:
    if task_name.lower() == "mnli":
        num_classes = 3
    elif task_name.lower() == "sts-b":
        num_classes = 1  
    elif task_name.lower() in ["mrpc", "qqp", "wnli", "rte", "qnli", "cola"]:
        num_classes = 2
    elif task_name.lower() in ["sst2"]:
        num_classes = 1
    else:
        raise ValueError(f"Unsupported task: {task_name}")
        
    in_channels = train_data.num_features

    # ----------------- Grid Search Setup -----------------
    # Define candidate values for grid search hyperparameters
    temperature_values = [0.3, 0.5, 0.7]
    pretrain_epoch_values = [5, 10]  
    finetune_epoch_values = [5, 10] 

    best_metric = -float('inf')
    best_config = None
    grid_results = []

    print("Starting grid search over hyperparameters...")
    for temp in temperature_values:
        for pretrain_epochs in pretrain_epoch_values:
            for finetune_epochs in finetune_epoch_values:
                print(f"\nConfiguration: Temperature={temp}, Pretrain Epochs={pretrain_epochs}, Fine-tune Epochs={finetune_epochs}")

                # Initialize a new model for this configuration
                model = GraphMaskedLM(in_channels, hidden_channels=512, num_classes=num_classes, nhead=4).to(device)
                try:
                    print(torchinfo.summary(model))
                except Exception as e:
                    print(f"Could not print model summary: {e}")
                model.encoder.gradient_checkpointing_enable()

                # Contrastive pretraining
                print("Starting contrastive pretraining...")
                pretrain_contrastive(model, train_data, pretrain_epochs=pretrain_epochs, temp=temp)

                # Fine-tuning on GLUE task
                print(f"Fine-tuning on {task_name}...")
                train_glue(model, train_data, train_texts, train_labels, task_name, num_epochs=finetune_epochs)

                # Evaluation on test set
                if test_data is not None:
                    print(f"Evaluating on {task_name} test set...")
                    test_data = test_data.to(device)
                    results_eval = evaluate_glue(model, test_data, test_texts, test_labels, task_name)
                    if task_name.lower() == "cola":
                        metric = results_eval.get("matthews_correlation", 0)
                    elif task_name.lower() == "sts-b":
                        # Use average of Pearson and Spearman as the metric for STS-B
                        metric = results_eval.get("avg_correlation", 0)
                    else:
                        metric = results_eval.get("accuracy", 0)

                print(f"Result for configuration (Temp={temp}, Pretrain_Epochs={pretrain_epochs}, Fine-tune_Epochs={finetune_epochs}): Metric={metric}")
                grid_results.append((temp, pretrain_epochs, finetune_epochs, metric))
                if metric > best_metric:
                    best_metric = metric
                    best_config = (temp, pretrain_epochs, finetune_epochs)

                # Free memory after each run
                torch.cuda.empty_cache()
                gc.collect()

    print("\nGrid Search Completed.")
    print("Grid Search Results:")
    for config in grid_results:
        print(f"Temperature={config[0]}, Pretrain Epochs={config[1]}, Fine-tune Epochs={config[2]} --> Metric={config[3]}")

    if best_config is not None:
        print(f"\nBest configuration: Temperature={best_config[0]}, Pretrain Epochs={best_config[1]}, Fine-tune Epochs={best_config[2]} with Metric={best_metric}")
        # Re-train best model to save
        print("Retraining best model...")
        best_model = GraphMaskedLM(in_channels, hidden_channels=128, num_classes=num_classes, nhead=4).to(device)
        try:
            print(torchinfo.summary(best_model))
        except Exception as e:
            print(f"Could not print model summary: {e}")
        best_model.encoder.gradient_checkpointing_enable()
        pretrain_contrastive(best_model, train_data, pretrain_epochs=best_config[1], temp=best_config[0])
        train_glue(best_model, train_data, train_texts, train_labels, task_name, num_epochs=best_config[2])
        if test_data is not None:
            results_eval = evaluate_glue(best_model, test_data, test_texts, test_labels, task_name)
            print("Best model evaluation results:", results_eval)
        torch.save(best_model.state_dict(), f"graph_mlm_{task_name}_best_model.pt")
        print("Saved best model as graph_mlm_{}_best_model.pt".format(task_name))
    else:
        print("No valid configuration found.")

    print("Done!")
    torch.cuda.empty_cache()
    gc.collect()
