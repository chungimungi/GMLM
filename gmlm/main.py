import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from models import GraphMaskedLM
from train import pretrain_contrastive, train_model
from evaluation import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_configs = [
    {'name': 'Amazon-Photo', 'class': Amazon, 'kwargs': {'root': './data/Amazon-Photo', 'name': 'Photo'}, 'split_transform': True},
    {'name': 'Amazon-Computers', 'class': Amazon, 'kwargs': {'root': './data/Amazon-Computers', 'name': 'Computers'}, 'split_transform': True},
    {'name': 'Coauthor-CS', 'class': Coauthor, 'kwargs': {'root': './data/Coauthor-CS', 'name': 'CS'}, 'split_transform': True},
    {'name': 'Cora', 'class': Planetoid, 'kwargs': {'root': './data', 'name': 'Cora'}, 'split_transform': True},
    {'name': 'Citeseer', 'class': Planetoid, 'kwargs': {'root': './data', 'name': 'Citeseer'}, 'split_transform': True},
    {'name': 'PubMed', 'class': Planetoid, 'kwargs': {'root': './data', 'name': 'PubMed'}, 'split_transform': True},
]

for config in dataset_configs:
    dataset_name = config['name']
    dataset_class = config['class']
    kwargs = config['kwargs']
    split_transform = config['split_transform']

    print(f"\n=== Running on {dataset_name} dataset ===")

    transforms = [T.NormalizeFeatures()]
    if split_transform:
        transforms.append(T.RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2))

    dataset = dataset_class(transform=T.Compose(transforms), **kwargs)
    data = dataset[0].to(device)

    in_channels = dataset.num_features
    num_classes = dataset.num_classes
    model = GraphMaskedLM(in_channels, hidden_channels=128, num_classes=num_classes, nhead=4).to(device)

    print("Starting contrastive pretraining...")
    pretrain_contrastive(model, data, pretrain_epochs=20, temp=0.5)

    print("Starting fine-tuning...")
    train_model(model, data, num_epochs=150)

    print("Evaluating model...")
    evaluate_model(model, data)

print("\n=== All datasets processed successfully! ===")
