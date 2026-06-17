from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import get_mnist_data
from model import DigitClassifier

MODEL_PATH = Path(__file__).parent / "models" / "digit_classifier.pt"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "inference_batch.png"

_, _, test_loader = get_mnist_data(batch_size=40)
images, labels = next(iter(test_loader))

model = DigitClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

with torch.no_grad():
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

print(f"Images shape: {images.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Predictions shape: {predictions.shape}")

fig, axes = plt.subplots(4, 10, figsize=(14, 6))

for i, ax in enumerate(axes.flat):
    image = images[i].squeeze() * 0.3081 + 0.1307
    true_label = labels[i].item()
    predicted_label = predictions[i].item()
    confidence = probabilities[i, predicted_label].item()

    ax.imshow(image, cmap="gray")

    color = "green" if predicted_label == true_label else "red"
    ax.set_title(
        f"True: {true_label}\nPred: {predicted_label} ({confidence:.2f})",
        color=color,
    )
    ax.axis("off")

plt.tight_layout()
OUTPUT_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved inference image to {OUTPUT_PATH}")

plt.show()