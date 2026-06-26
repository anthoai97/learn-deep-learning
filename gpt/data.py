import json
from pathlib import Path
from datasets import load_dataset
import torch

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.pt"
VAL_PATH = PROCESSED_DIR / "val.pt"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"

# Objective: Data pipeline loading dataset

def load_tinystores():
  dataset = load_dataset("roneneldan/TinyStories", cache_dir=DATA_DIR)
  return dataset

def build_char_tokenizer_from_chars(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(indices):
        return "".join(itos[i] for i in indices)

    return encode, decode, stoi, itos

def build_char_tokenizer(text):
    chars = sorted(set(text))
    return build_char_tokenizer_from_chars(chars)

def get_batch(data, batch_size, context_length):
    ix = torch.randint(len(data) - context_length - 1, (batch_size,))

    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])

    return x, y

def preprocess_data(num_stories=1000):
    PROCESSED_DIR.mkdir(exist_ok=True)

    dataset = load_tinystores()

    texts = [example["text"] for example in dataset["train"].select(range(num_stories))]
    train_text = "\n".join(texts)

    encode, decode, stoi, _ = build_char_tokenizer(train_text)

    ids = encode(train_text)
    data = torch.tensor(ids, dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    chars = sorted(stoi.keys())

    torch.save(train_data, TRAIN_PATH)
    torch.save(val_data, VAL_PATH)

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump({"chars": chars}, f, ensure_ascii=False, indent=2)

    return train_data, val_data, encode, decode, len(chars)

def prepare_data(num_stories = 3000, force_preprocess=False):
    if force_preprocess or not TRAIN_PATH.exists() or not VAL_PATH.exists() or not VOCAB_PATH.exists():
        return preprocess_data(num_stories)

    train_data = torch.load(TRAIN_PATH)
    val_data = torch.load(VAL_PATH)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    chars = vocab["chars"]
    encode, decode, stoi, _ = build_char_tokenizer_from_chars(chars)

    return train_data, val_data, encode, decode, len(stoi)

def main():
    train_data, val_data, encode, decode, vocab_size = prepare_data()

    batch_size = 4
    context_length = 8

    x, y = get_batch(train_data, batch_size, context_length)

    print("vocab size:", vocab_size)
    print("train data shape:", train_data.shape)
    print("val data shape:", val_data.shape)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("decoded x[0]:", decode(x[0].tolist()))
    print("decoded y[0]:", decode(y[0].tolist()))

if __name__ == "__main__":
  main()
