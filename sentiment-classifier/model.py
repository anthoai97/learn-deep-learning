import torch
from torch import nn

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_dim=2):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids): # (32, seq_len)
        embedded = self.embedding(input_ids) # (32, seq_len, 64)
        
        output, _ = self.lstm(embedded)
        
        lengths = (input_ids != 0).sum(dim=1)

        last_indices = lengths - 1

        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        final_hidden = output[batch_indices, last_indices]
        logits = self.classifier(final_hidden)

        return logits