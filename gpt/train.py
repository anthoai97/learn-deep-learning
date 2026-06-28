import time

from data import get_batch, prepare_data
from model import BigramLanguageModel
import torch

batch_size = 32
context_length = 64
n_embd = 64
num_heads = 4
n_layer = 4
head_size = 8
eval_iters = 100
max_steps = 3000

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

def print_training_config(
    batch_size,
    context_length,
    eval_iters,
    n_embd,
    num_heads,
    n_layer,
):
    head_size = n_embd // num_heads

    print("=====================")
    print("Training config")
    print("=====================")
    print(f"batch_size:     {batch_size}")
    print(f"context_length: {context_length}")
    print(f"eval_iters:     {eval_iters}")
    print(f"n_embd:         {n_embd}")
    print(f"num_heads:      {num_heads}")
    print(f"head_size:      {head_size}")
    print(f"n_layer:        {n_layer}")
    print("=====================")

@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    out = {}

    model.eval()

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length)
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def print_model_summary(model):
    total, trainable = count_parameters(model)

    # print(model)
    print("=====================")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

def main():
    train_data, val_data, encode, decode, vocab_size = prepare_data()

    device = device_info()
    
    model = BigramLanguageModel(vocab_size, context_length, n_embd, num_heads, n_layer)
    model.to(device)
    
    print_training_config(
        batch_size,
        context_length,
        eval_iters,
        n_embd,
        num_heads,
        n_layer,
    )
    
    print_model_summary(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    start_time = time.time()
    
    for step in range(max_steps):
        x, y = get_batch(train_data, batch_size, context_length)
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 300 == 0:
            losses = estimate_loss(model, train_data, val_data, device)
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    print("=====================")
    
            
    prompt = "Once"

    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=200)

    print(decode(generated[0].tolist()))

if __name__ == "__main__":
    main()

