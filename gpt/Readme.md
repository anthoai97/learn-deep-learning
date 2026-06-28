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


## Experiment Tracks

### 2026-06-28: 
#### Add attention block
step 700: train loss 2.5630, val loss 2.5883
step 800: train loss 2.5480, val loss 2.5735
step 900: train loss 2.5282, val loss 2.5617

Oncethud rlthe tut tosor sis phe wand pouy iwe and borivhep, th Iraivilroleaduthew. Le dilyt,.



Sed to. t int spppalago anroliinder os m te. tha hlre Tt The beigwosd Je t. hey co, ve at be. ca ely,e tib

#### Add Multi-head block
step 700: train loss 2.4075, val loss 2.4278
step 800: train loss 2.3914, val loss 2.4072
step 900: train loss 2.3722, val loss 2.4021
=====================
Once she sp ire hean dean fkes arireawagu ady ganssckkedapit arelllg sthay Shin, hed ire "he Therird, s may ss d an tond py he. bke hiciiet he thet. fy e were to Thily all oy ce kean n'tikeree s s ttat a

#### Add Feedforward Block
step 700: train loss 2.3649, val loss 2.3855
step 800: train loss 2.3447, val loss 2.3579
step 900: train loss 2.3218, val loss 2.3447
Total training time: 16.53 seconds
=====================
Once?erâ€€ d w thilgong! Shr thaisid theran. sh ka hanced the was o bthengig th they herto s, indimamoure m tthan hinildisatd g ad frr she tshig m! Fad cory hesedhenomrtored ssou ny'tit pd tosh e idps man

#### Add residual connection
Apple Silicon GPU is available
step 700: train loss 2.3162, val loss 2.3382
step 800: train loss 2.2930, val loss 2.3173
step 900: train loss 2.2751, val loss 2.2943
Total training time: 18.34 seconds
=====================
Oncent pay he gas thanatomay ntt Sle was ans bad !"Win. Tashingke sk, the d oky " tengead cf sed alle ng bed angond s w is theyscony hennanoy. somere ie nmuldey aird he ba heke thid re mazey shathe m. ad

#### Stack more Blocks
Total parameters: 54,170
Trainable parameters: 54,170
step 700: train loss 2.2035, val loss 2.2203
step 800: train loss 2.1604, val loss 2.1758
step 900: train loss 2.1236, val loss 2.1406
Total training time: 42.12 seconds
=====================
Onceys ait ma yout. The tot they pall fractou he thedil trleer the ceare nut beyole.

En, The pait to stouche stin. To dow the clisked.

The arong aye theruck nlums finurlasil.

"Cac wa bas and dovorid g

#### More
=====================
Training config
=====================
batch_size:     32
context_length: 64
eval_iters:     100
n_embd:         64
num_heads:      4
head_size:      16
n_layer:        4
=====================
=====================
Total parameters: 198,362
Trainable parameters: 198,362
step 0: train loss 4.4703, val loss 4.4741
step 300: train loss 2.2131, val loss 2.2351
step 600: train loss 1.8888, val loss 1.8996
step 900: train loss 1.6789, val loss 1.6734
step 1200: train loss 1.5338, val loss 1.5332
step 1500: train loss 1.4598, val loss 1.4518
step 1800: train loss 1.3956, val loss 1.3724
step 2100: train loss 1.3525, val loss 1.3307
step 2400: train loss 1.3182, val loss 1.3098
step 2700: train loss 1.2808, val loss 1.2838
Total training time: 94.26 seconds
=====================
Once wanted to she was everyone ball, found away felt ball got a boaking before.

"You pictined, he sunnay dance onteling flower things. You rest, they say he didn't the minal polizzantly. Ben went like t