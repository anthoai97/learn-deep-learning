import torch

def evaluate(model, data_loader, loss_fn, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item() * images.size(0)  # Multiply by batch size to get total loss
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples    

    return average_loss, accuracy