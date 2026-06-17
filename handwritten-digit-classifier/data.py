from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR = Path(__file__).parent / "data"

print(f"DATA_DIR: {DATA_DIR}")

def get_mnist_data(batch_size: int = 32):
	print("Getting MNIST data...")

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	full_train_dataset = datasets.MNIST(
		root=DATA_DIR,
		train=True,
		download=True,
		transform=transform,
	)

	test_dataset = datasets.MNIST(
		root=DATA_DIR,
		train=False,
		download=True,
		transform=transform,
	)

	train_dataset, val_dataset = random_split(
		full_train_dataset, [50000, 10000]
	)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, test_loader
