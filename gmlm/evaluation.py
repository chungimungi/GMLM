import torch

def evaluate_model(model, data):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        test_mask = data.test_mask
        logits = model(data.x, data.edge_index, test_mask)
        loss = criterion(logits[test_mask], data.y[test_mask])
        
        pred = logits[test_mask].max(dim=1)[1]
        acc = pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
        
        print(f"Test Loss: {loss.item():.4f}, Accuracy: {acc*100:.2f}%")
