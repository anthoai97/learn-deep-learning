## Objective

Build a movie review sentiment classifier that outputs Positive/Negative with probability.

Works:
- [x] Load dataset - IMDb Reviews (Hugging Face): https://huggingface.co/datasets/stanfordnlp/imdb
- [x] Tokenization, vocabuluary, embedding
- [x] Network
- [x] Train - Val - Test
- [x] Inference
- [] Optimize - Architect, Train Loop, ...


## Commands

Train the model:

```bash
uv run --package sentiment-classifier python sentiment-classifier/train.py
```

Run inference and save the prediction image:

```bash
uv run --package sentiment-classifier python sentiment-classifier/inference.py
```

## Experiment tracks
#### 24-06-2026: Train with more epochs:
- 1 epoch
```
Train Loss: 0.6580, Val Loss: 0.5904, Val Accuracy: 0.7832
Test Accuracy: 0.7667
```

- 5 epocs
```text
Train Loss: 0.2730, Val Loss: 0.3034, Val Accuracy: 0.8904
Total training time: 437.59 seconds
Test Accuracy: 0.8783
```

- 10 epocs
```text
Train Loss: 0.2730, Val Loss: 0.3034, Val Accuracy: 0.8904
Total training time: 437.59 seconds
Test Accuracy: 0.8783
```