import torch
from data import get_mnist_data
import matplotlib.pyplot as plt
from model import DigitClassifier
from torch import nn
from eval import evaluate
from pathlib import Path
import time

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "digit_classifier.pt"

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
	total_loss = 0
	model.train() # Tells PyTorch we are in training mode, which can affect layers like dropout and batchnorm
	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)

		output = model(images)
		loss = loss_fn(output, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	return total_loss / len(train_loader)
	
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

def main():
	print("Hello from handwritten-digit-classifier!")
	train_loader, val_loader, test_loader = get_mnist_data(batch_size=32)
	images, labels = next(iter(train_loader))

	print(images.shape) # ([32, 1, 28, 28]
	print(labels.shape) # ([32])

	# image = images[1]
	# label = labels[1]

	# plt.imshow(image.squeeze(), cmap="gray")
	# plt.title(f"Label: {label.item()}")
	# plt.axis("off")
	# plt.show()

	device = device_info()
	
	print(f"Using device: {device}")

	model = DigitClassifier() # output = model(images) # ([32, 10])
	model = model.to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	epochs = 20
	start_time = time.time()
	print("Starting training loop...")
	for epoch in range(epochs):
		avg_loss = train_one_epoch(
			model,
			train_loader,
			loss_fn,
			optimizer,
			device
		)

		val_loss, val_accuracy = evaluate(
			model,
			val_loader,
			loss_fn,
			device
		)

		print(
			f"Epoch [{epoch+1}/{epochs}], "
			f"Train Loss: {avg_loss:.4f}, "
			f"Val Loss: {val_loss:.4f}, "
			f"Val Accuracy: {val_accuracy:.4f}"
		)

	print("Training complete!")
	print(f"Total training time: {time.time() - start_time:.2f} seconds")
	print("Evaluating on test set...")

	test_loss, test_accuracy = evaluate(
		model,
		test_loader,
		loss_fn,
		device
	)

	print(
		f"Test Loss: {test_loss:.4f}, "
		f"Test Accuracy: {test_accuracy:.4f}"
	)

	MODEL_DIR.mkdir(exist_ok=True)
	torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
		main()	