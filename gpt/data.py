from pathlib import Path
from datasets import load_dataset
import torch

DATA_DIR = Path(__file__).parent / "data"

# Objective: Data pipeline loading dataset

def load_tinystores():
  dataset = load_dataset("roneneldan/TinyStories", cache_dir=DATA_DIR)
  return dataset

def build_char_tokenizer(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(indices):
        return "".join(itos[i] for i in indices)

    return encode, decode, stoi, itos

def get_batch(data, batch_size, context_length):
    ix = torch.randint(len(data) - context_length - 1, (batch_size,))

    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])

    return x, y

def main():
    DATA_DIR.mkdir(exist_ok=True)
    dataset = load_tinystores()

    texts = [example["text"] for example in dataset["train"].select(range(1000))]
    train_text = '\n'.join(texts)

    encode, decode, stoi, itos = build_char_tokenizer(train_text)

    ids = encode(train_text)
    data = torch.tensor(ids, dtype=torch.long)
    print(f"data shape: {data.shape}")
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    batch_size = 4
    context_length = 8

    x, y = get_batch(train_data, batch_size, context_length)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x[0]:", x[0])
    print("y[0]:", y[0])
    print("decoded x[0]:", decode(x[0].tolist()))
    print("decoded y[0]:", decode(y[0].tolist()))



if __name__ == "__main__":
  main()
