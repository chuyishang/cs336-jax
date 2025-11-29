# Complete Training Workflow

This guide shows the complete workflow from raw text data to training on Modal GPUs.

## Overview

```
Raw Text Files → Tokenize → Binary Files → Upload to Modal → Train on GPU
```

## Step-by-Step Guide

### Step 1: Prepare Your Data (Tokenization)

Your text files need to be tokenized and converted to binary format first.

```bash
# Train BPE tokenizer and tokenize both datasets
uv run python prepare_data.py \
  --train-path data/TinyStoriesV2-GPT4-train.txt \
  --val-path data/TinyStoriesV2-GPT4-valid.txt \
  --vocab-size 10000 \
  --output-dir data \
  --tokenizer-dir tokenizer
```

This will:
- Train a BPE tokenizer with 10,000 tokens
- Tokenize your training data → `data/train.bin`
- Tokenize your validation data → `data/val.bin`
- Save tokenizer to `tokenizer/vocab.pkl` and `tokenizer/merges.pkl`

**Expected output:**
```
================================================================================
Training BPE Tokenizer
================================================================================
Training data: data/TinyStoriesV2-GPT4-train.txt
Vocab size: 10000
Special tokens: ['<|endoftext|>']

Training BPE (this may take a while for large files)...
✓ Training complete!
  Vocabulary size: 10000
  Number of merges: 9743

✓ Tokenizer saved to tokenizer/
  - vocab.pkl
  - merges.pkl

================================================================================
Tokenizing Training Data
================================================================================
Tokenizing data/TinyStoriesV2-GPT4-train.txt...
File size: 2048.00 MB
✓ Tokenization complete!
  Total tokens: 587,432,123
✓ Saved to data/train.bin
  Output size: 1117.45 MB
  Compression ratio: 1.83x
```

### Step 2: Setup Modal (One-time)

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal setup

# Create persistent volume
modal volume create transformer-training-vol
```

### Step 3: Upload Data to Modal

```bash
# Upload tokenized training data
modal run cs336_basics/modal_utils.py \
  --action upload \
  --local-path data/train.bin \
  --remote-name train.bin \
  --data-type data

# Upload tokenized validation data
modal run cs336_basics/modal_utils.py \
  --action upload \
  --local-path data/val.bin \
  --remote-name val.bin \
  --data-type data
```

**Note:** For large files (>1GB), this may take several minutes.

### Step 4: Configure Training

Edit `cs336_basics/config.yaml` if needed:

```yaml
# Model Architecture
model:
  vocab_size: 10000  # Must match your tokenizer vocab size!
  d_model: 512
  num_heads: 8
  num_layers: 6
  context_length: 1024

# Training
training:
  batch_size: 32
  max_iters: 100000

# Data (already configured)
data:
  train_path: train.bin  # Binary tokenized data
  val_path: val.bin
```

**Important:** Make sure `vocab_size` in config matches your tokenizer!

### Step 5: Train on Modal

```bash
# Basic training
modal run cs336_basics/train_modal.py

# With Weights & Biases logging
modal run cs336_basics/train_modal.py \
  --wandb \
  --run-name my-experiment

# With custom config
modal run cs336_basics/train_modal.py \
  --config-path cs336_basics/config.yaml \
  --wandb \
  --run-name tinystories-10k-vocab
```

### Step 6: Monitor Training

- **Console:** Real-time logs in your terminal
- **Modal Dashboard:** https://modal.com/apps
- **Weights & Biases:** https://wandb.ai (if enabled)

### Step 7: Download Checkpoints

```bash
# List available checkpoints
modal run cs336_basics/modal_utils.py \
  --action list \
  --data-type checkpoint

# Download final model
modal run cs336_basics/modal_utils.py \
  --action download \
  --remote-name checkpoint_final.pt \
  --local-path checkpoints/checkpoint_final.pt \
  --data-type checkpoint
```

## Quick Reference Commands

### Data Preparation

```bash
# Full tokenization pipeline
uv run python prepare_data.py \
  --train-path data/TinyStoriesV2-GPT4-train.txt \
  --val-path data/TinyStoriesV2-GPT4-valid.txt \
  --vocab-size 10000

# Use existing tokenizer (if already trained)
uv run python prepare_data.py \
  --train-path data/new_data.txt \
  --vocab-size 10000 \
  --use-existing-tokenizer
```

### Modal Operations

```bash
# Upload data
modal run cs336_basics/modal_utils.py \
  --action upload \
  --local-path data/train.bin \
  --remote-name train.bin \
  --data-type data

# List files
modal run cs336_basics/modal_utils.py --action list --data-type data
modal run cs336_basics/modal_utils.py --action list --data-type checkpoint

# Download checkpoint
modal run cs336_basics/modal_utils.py \
  --action download \
  --remote-name checkpoint_final.pt \
  --local-path checkpoints/checkpoint_final.pt

# Delete old checkpoint
modal run cs336_basics/modal_utils.py \
  --action delete \
  --remote-name checkpoint_iter_5000.pt
```

### Training

```bash
# Local training (if you have GPU)
uv run python cs336_basics/train.py

# Modal training
modal run cs336_basics/train_modal.py --wandb --run-name experiment-1
```

## Common Issues and Solutions

### Issue: "vocab_size mismatch"

**Problem:** Your config vocab_size doesn't match tokenizer vocab_size

**Solution:**
```bash
# Check your tokenizer vocab size
uv run python -c "import pickle; print(len(pickle.load(open('tokenizer/vocab.pkl', 'rb'))))"

# Update config.yaml to match
```

### Issue: "Data file not found on Modal"

**Problem:** Forgot to upload data files

**Solution:**
```bash
# Upload both train and val files
modal run cs336_basics/modal_utils.py --action upload --local-path data/train.bin --remote-name train.bin --data-type data
modal run cs336_basics/modal_utils.py --action upload --local-path data/val.bin --remote-name val.bin --data-type data
```

### Issue: "Out of memory during training"

**Problem:** Batch size or model too large for GPU

**Solution:** Edit `config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32

model:
  d_model: 256    # Reduce from 512
  num_layers: 4   # Reduce from 6
```

Or use larger GPU in `train_modal.py`:
```python
@app.function(
    gpu="A100",  # Upgrade from A10G
    ...
)
```

### Issue: "Tokenization takes too long"

**Problem:** Large files take time to process

**Solution:** The script processes in chunks. For very large files:
- Use a smaller vocab size for faster training
- Process on a machine with more RAM
- Consider sampling your data first for testing

## Best Practices

1. **Start Small:** Test with small vocab (1000) and few iterations (1000) first
2. **Monitor Costs:** Check Modal dashboard for GPU usage
3. **Save Tokenizer:** Keep your tokenizer files safe - you'll need them for inference
4. **Version Control:** Track your config.yaml changes with git
5. **Checkpoints:** Save frequently enough to recover from interruptions

## Complete Example Workflow

```bash
# 1. Prepare data
uv run python prepare_data.py \
  --train-path data/TinyStoriesV2-GPT4-train.txt \
  --val-path data/TinyStoriesV2-GPT4-valid.txt \
  --vocab-size 10000

# 2. Setup Modal (first time only)
modal setup
modal volume create transformer-training-vol

# 3. Upload data
modal run cs336_basics/modal_utils.py --action upload --local-path data/train.bin --remote-name train.bin --data-type data
modal run cs336_basics/modal_utils.py --action upload --local-path data/val.bin --remote-name val.bin --data-type data

# 4. Update config (check vocab_size matches!)
# Edit cs336_basics/config.yaml

# 5. Train
modal run cs336_basics/train_modal.py --wandb --run-name tinystories-experiment

# 6. Download results
modal run cs336_basics/modal_utils.py --action download --remote-name checkpoint_final.pt --local-path checkpoints/checkpoint_final.pt
```

## Cost Estimates

For TinyStories dataset (~2GB text, ~1GB tokenized):
- **Tokenization:** Free (local)
- **Upload:** Free (Modal includes bandwidth)
- **Training (100k iters, A10G):**
  - Time: ~10-20 hours
  - Cost: ~$11-22
- **Download:** Free

## Next Steps

After training:
1. Download your checkpoint
2. Use the tokenizer for inference
3. Evaluate on test set
4. Fine-tune if needed

See `TRAINING_GUIDE.md` for local training or `MODAL_GUIDE.md` for advanced Modal usage.
