import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, n_embd):
        super().__init__()

        self.context_length = context_length

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: [B, T]
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        # token_emb shape: [B, T, n_embd]

        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)
        # pos_emb shape: [T, n_embd]

        x = token_emb + pos_emb
        # x shape: [B, T, n_embd]

        logits = self.lm_head(x)
        # logits shape: [B, T, vocab_size]

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_for_loss = logits.reshape(B * T, C)
            targets_for_loss = targets.reshape(B * T)

            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            # idx_cond shape: [B, <=context_length]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]
            # logits shape: [B, vocab_size]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            # idx_next shape: [B, 1]

            idx = torch.cat((idx, idx_next), dim=1)

        return idx