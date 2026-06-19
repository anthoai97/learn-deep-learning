from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import get_sentiment_loaders
from model import SentimentClassifier


MODEL_PATH = Path(__file__).parent / "models" / "sentiment-classifer.pt"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "inference_batch.png"
REPORT_PATH = OUTPUT_DIR / "sentiment_predictions.txt"

_, _, test_loader, vocab = get_sentiment_loaders(batch_size=32)

model = SentimentClassifier(vocab_size=len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

label_names = ["negative", "positive"]

# 1. Run all test data for accuracy
total_correct = 0
total_samples = 0

with torch.no_grad():
    for input_ids, labels in test_loader:
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

test_accuracy = total_correct / total_samples
print(f"Test Accuracy: {test_accuracy:.4f}")

# 2. Visualize only 20 samples
visual_input_ids, visual_labels = next(iter(test_loader))
visual_input_ids = visual_input_ids[:20]
visual_labels = visual_labels[:20]

with torch.no_grad():
    visual_logits = model(visual_input_ids)
    visual_probabilities = torch.softmax(visual_logits, dim=1)
    visual_predictions = visual_probabilities.argmax(dim=1)

raw_texts = [
    test_loader.dataset.data[i]["text"]
    for i in range(20)
]

colors = [
    "green" if visual_predictions[i].item() == visual_labels[i].item() else "red"
    for i in range(20)
]

confidences = [
    visual_probabilities[i, visual_predictions[i]].item()
    for i in range(20)
]

titles = [
    f"{i}: {label_names[visual_predictions[i].item()]}"
    for i in range(20)
]

report_lines = []

for i in range(20):
    true_label = label_names[visual_labels[i].item()]
    predicted_label = label_names[visual_predictions[i].item()]
    confidence = confidences[i]
    snippet = raw_texts[i].replace("<br />", " ")[:120]

    sample_report = (
        f"{'=' * 80}\n"
        f"Review:     {snippet}...\n"
        f"True:       {true_label}\n"
        f"Predicted:  {predicted_label}\n"
        f"Confidence: {confidence:.4f}\n"
    )

    print(sample_report)
    report_lines.append(sample_report)

plt.figure(figsize=(10, 8))
plt.barh(titles, confidences, color=colors)
plt.xlabel("Confidence")
plt.title("Sentiment predictions on 20 test samples")
plt.xlim(0, 1)
plt.tight_layout()
OUTPUT_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
REPORT_PATH.write_text("\n".join(report_lines))
print(f"Saved inference image to {OUTPUT_PATH}")
plt.show()