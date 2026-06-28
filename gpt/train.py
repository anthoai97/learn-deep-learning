from data import get_batch, prepare_data
from model import BigramLanguageModel
import torch

batch_size = 32
context_length = 64
eval_iters = 100
n_embd = 32

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

def main():
    train_data, val_data, encode, decode, vocab_size = prepare_data()

    device = device_info()
    
    model = BigramLanguageModel(vocab_size, context_length, n_embd)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in range(1000):
        x, y = get_batch(train_data, batch_size, context_length)
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            losses = estimate_loss(model, train_data, val_data, device)
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            
    prompt = "Once"

    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=200)

    print(decode(generated[0].tolist()))

if __name__ == "__main__":
    main()

