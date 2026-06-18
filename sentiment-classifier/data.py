from collections import Counter
from pathlib import Path
import string
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).parent / "data"
print(f"DATA_DIR: {DATA_DIR}")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def get_sentiment_data():
    print("Getting sentiment data...")

    # Load the dataset
    dataset = load_dataset("stanfordnlp/imdb", cache_dir=DATA_DIR)
    train_data = dataset["train"]
    test_data = dataset["test"]

    # print(train_data[0])

    return train_data, test_data

# Tokenizer
def tokenize_text(text: str) -> list[str]:
    # Simple whitespace tokenizer
    text = text.lower()  # Convert to lowercase
    text = text.replace("<br />", " ")

    for punctuation in string.punctuation:
        text = text.replace(punctuation, " ")
    tokens = text.split()
    return tokens

def build_vocab(dataset, max_vocab_size: int = 20000) -> dict[str, int]:
    counter = Counter()
    
    for item in dataset:
        tokens = tokenize_text(item["text"])
        counter.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    most_common = counter.most_common(max_vocab_size - len(vocab))
    for token, _ in most_common:
        vocab[token] = len(vocab)

    return vocab

def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    tokens = tokenize_text(text)
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

def decode_text(encoded: list[int], vocab: dict[str, int]) -> str:
    inv_vocab = {idx: token for token, idx in vocab.items()}
    tokens = [inv_vocab.get(idx, UNK_TOKEN) for idx in encoded]
    return " ".join(tokens)

def collate_batch(batch):
    input_ids, labels = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return input_ids, labels
    
class SentimentDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]
        input_ids = encode_text(text, self.vocab)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, label

def get_sentiment_loaders( batch_size: int = 4):
    train_data, test_data = get_sentiment_data()
    vocab = build_vocab(train_data, max_vocab_size=20000)

    full_train_dataset = SentimentDataset(train_data, vocab)
    test_dataset = SentimentDataset(test_data, vocab)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    return train_loader, val_loader, test_loader, vocab

if __name__ == "__main__":
    train_data, test_data = get_sentiment_data()
    print(f"Train data: {train_data}")
    print(f"Test data: {test_data}")