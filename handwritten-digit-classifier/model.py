import torch
from torch import nn

class DigitClassifier(nn.Module):
	def __init__(self):
		super(DigitClassifier, self).__init__()
		self.flatten = nn.Flatten()
		self.network = nn.Sequential(
			nn.Linear(28 * 28, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 10)
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.network(x)
		return logits