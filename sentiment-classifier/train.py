from pathlib import Path

from data import get_sentiment_loaders
from model import SentimentClassifier
from tqdm import tqdm
import torch
from torch import nn
from evaluate import evaluate

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "sentiment-classifer.pt"

def device_info():
    device = "cpu"
    if torch.cuda.is_available():
        print("CUDA is available")
        print(torch.cuda.get_device_name(0))
        device = "cuda"
    elif torch.backends.mps.is_available():
        print("Apple Silicon GPU is available")
        device = "mps"
    else:
        print("Using CPU")
    return device

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    total_loss = 0
    print("Debug")
    model.train()

    for input_ids, labels in tqdm(train_loader):
        input_ids, labels = input_ids.to(device), labels.to(device)

        output = model(input_ids)
        loss = loss_fn(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def main():
    print("Hello from sentiment-classifier!")

    train_loader, val_loader, test_loader, vocab = get_sentiment_loaders(batch_size=32)

    input_ids, labels = next(iter(train_loader))

    print(f"Input ids shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    device = device_info()

    model = SentimentClassifier(vocab_size=len(vocab))
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    avg_loss = train_one_epoch(
			model,
			train_loader,
			loss_fn,
			optimizer,
			device
		)
    
    val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

    print(
		f"Train Loss: {avg_loss:.4f}, "
		f"Val Loss: {val_loss:.4f}, "
		f"Val Accuracy: {val_accuracy:.4f}"
	)
    
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
	main()	