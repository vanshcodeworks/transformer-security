# Transformer Security

A character-level transformer encoder trained to classify real-world
vulnerability records, built from scratch in PyTorch.

## Layout

```
cyber.py                  model definition: embeddings, positional encoding,
                          transformer encoder blocks
trasformer_refrence.py    reference implementation the model is based on
train.py                  training loop
predict.py                inference
vocab.json                tokenizer vocabulary
```

## Setup

```bash
pip install torch
```

## Running

```bash
python train.py     # trains and writes best_model.pt
python predict.py   # loads the trained model and classifies
```

## Data

The training corpus is `real_world_vuln_data.csv` - scraped vulnerability
records. It is **not committed** (~44 MB). Place it alongside the scripts and
`train.py` will pick it up.

## Trained weights

`best_model.pt` is also **not committed** (~175 MB, comfortably over what a
repository should carry). Retrain with `train.py` to regenerate it.

`vocab.json` *is* committed - it is small, and inference cannot run without it.

## Notes

- The model is character-level, so the vocabulary is built from the character
  set of the corpus rather than from subword units.
- `trasformer_refrence.py` keeps the original reference implementation around
  for comparison; it is spelled that way in the filename.