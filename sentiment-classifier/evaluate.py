import torch
from tqdm import tqdm

def evaluate(model, val_loader, loss_fun, device):
    total_loss = 0
    correct = 0
    total = 0
    
    model.eval()
    
    with torch.no_grad():
        for input_ids, labels in tqdm(val_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            logits = model(input_ids)
            loss = loss_fun(logits, labels)
            
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    avg_loss = total_loss / len(val_loader)
    acc = correct / total
    
    return avg_loss, acc
