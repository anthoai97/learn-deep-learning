import torch
from torch import nn

class DigitClassifier(nn.Module):
	def __init__(self):
		super(DigitClassifier, self).__init__()

		self.flatten = nn.Flatten()
		self.linear = nn.Linear(28 * 28, 10)
			
	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear(x)
		return logits