# GPT Tiny
Train a 20M GPT from scratch on TinyStories using PyTorch.
Form Paper - https://arxiv.org/pdf/2305.07759

# Objectives
Pretrain a **tiny decoder-only Transformer (tiny GPT)** on short stories and generate continuations.

## Dataset

- TinyStories (Hugging Face): https://huggingface.co/datasets/roneneldan/TinyStories

## Architecture

`Tokenizer → Token embeddings → Positional embeddings → Causal Transformer blocks → Vocabulary logits → Next-token generation`

## Must-have features

- Tokenizer (BPE or character-level)
- Causal attention mask
- Training + validation loss
- Text generation
- Sampling: temperature + top-k
- Checkpoint saving/loading
- Inference engine
- Evaluation Pipeline

## Suggested starting config

- Layers: 4
- Heads: 4
- Embedding dim: 256
- Context length: 128–256
- Params: ~10M–30M

## Definition of done

- Train/val curves saved
- Generate samples from prompts like:
    
    `Once upon a time, a little dog`

## A good learning sequence would be:
1. Data pipeline: load TinyStories, tokenize text, create train/val splits.
2. Batch creation: convert token streams into (x, y) pairs where y is x shifted by one token.
3. Minimal GPT model: embeddings, positional embeddings, one causal self-attention block, then stack blocks.
4. Training loop: cross-entropy next-token loss, validation loss, checkpoints.
5. Generation: autoregressive loop with temperature and top-k sampling.
6. Curves and sample outputs.

# CMD

```bash
 uv run --package gpt python .\gpt\data.py
```
