import torch
from data import get_mnist_data
import matplotlib.pyplot as plt
from model import DigitClassifier
from torch import nn
import time

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
	total_loss = 0
	model = model.to(device)
	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)

		output = model(images)
		loss = loss_fn(output, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	return total_loss / len(train_loader)
	
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

	device = "cpu"
	if torch.cuda.is_available():
		print(torch.cuda.get_device_name(0))
		device = "cuda"
	else:
		print("CUDA GPU not available")
	
	print(f"Using device: {device}")

	model = DigitClassifier() # output = model(images) # ([32, 10])
	model = model.to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	epochs = 10
	print("Starting training loop...")
	for epoch in range(epochs):
		avg_loss = train_one_epoch(
			model,
			train_loader,
			loss_fn,
			optimizer,
			device
		)

		print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
		main()	