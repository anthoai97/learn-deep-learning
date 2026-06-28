import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size, context_length):
        super().__init__()
        
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_length, context_length))
        )
        
    def forward(self, x):
        # x shape: [B, T, n_embede]
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # k, q, v shape: [B, T, head_size]
        
        wei = q @ k.transpose(-2, -1) # [B, T, head_size] *[B, head_size, T]
        # wei shape: [B, T, T]
        
        wei = wei * (k.shape[-1] ** -0.5)
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # block future positions

        wei = F.softmax(wei, dim=-1)
        # each row now sums to 1

        out = wei @ v
        # out shape: [B, T, head_size]
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, context_length):
        super().__init__()
        
        assert n_embd % num_heads == 0
        
        head_size = n_embd // num_heads
        
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, context_length)
            for _ in range(num_heads)
        ])
        
    def forward(self, x):
        # each head output shape: [B, T, head_size]
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # out shape: [B, T, n_embd]
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, num_heads, context_length):
        super().__init__()
        
        self.sa = MultiHeadAttention(n_embd, num_heads, context_length)
        self.ffwd = FeedForward(n_embd)
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, n_embd, num_heads, n_layer):
        super().__init__()

        self.context_length = context_length

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        
        self.blocks = nn.Sequential(*[
            Block(n_embd, num_heads, context_length)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)

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
        
        x = self.blocks(x)
        # x shape: [B, T, n_embd]
        
        x = self.ln_f(x)
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