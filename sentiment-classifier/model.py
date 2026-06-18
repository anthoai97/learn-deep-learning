import torch
from torch import nn

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, output_dim=2):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        
        self.classifier = nn.Linear(embedding_dim, output_dim)

    def forward(self, input_ids): # (32, seq_len)
        embedded = self.embedding(input_ids) # (32, seq_len, 64)

        pooled = embedded.mean(dim=1) # (32, 64)
        
        logits = self.classifier(pooled) # (32, 2)

        return logits