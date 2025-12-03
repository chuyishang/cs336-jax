import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
import jax
from jax import Array
import jax.numpy as jnp
from flax import nnx
from flax.nnx import State
from jaxtyping import Float, Int

from cs336_jax_basics import model as model_module


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(use_wandb: bool, config: dict, run_name: Optional[str] = None):
    """Setup logging, optionally with Weights & Biases."""
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.get('wandb_project', 'cs336-assignment-1'),
                name=run_name,
                config=config
            )
            return wandb
        except ImportError:
            print("Warning: wandb not installed. Install with 'pip install wandb'")
            return None
    return None


class MemMapDataset:
    """Memory-efficient dataset using np.memmap for large datasets."""
    def __init__(self, data_path: str):
        """
        Initialize dataset with memory-mapped file.

        Args:
            data_path: Path to binary file containing tokenized data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        print(f"Loaded dataset from {data_path} with {len(self.data):,} tokens")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_batch_from_memmap(
    rngs: nnx.Rngs,
    dataset: MemMapDataset,
    batch_size: int,
    context_length: int,
) -> tuple[Array, Array]:
    """
    Sample a batch from memory-mapped dataset.

    Args:
        dataset: MemMapDataset instance
        batch_size: Number of sequences in batch
        context_length: Length of each sequence

    Returns:
        Tuple of (inputs, targets) tensors
    """
    # Use the model's get_batch function with the memmap data
    return model_module.get_batch(
        rngs,
        dataset.data,
        batch_size,
        context_length,
    )


@torch.no_grad()
def estimate_loss(
    rngs: nnx.Rngs,
    model: nnx.Module,
    train_dataset: MemMapDataset,
    val_dataset: Optional[MemMapDataset],
    config: dict,
    eval_iters: int
) -> dict:
    """
    Estimate loss on training and validation sets.

    Args:
        model: The model to evaluate
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Configuration dictionary
        eval_iters: Number of iterations to evaluate

    Returns:
        Dictionary with 'train' and 'val' losses
    """
    losses = {}

    batch_size = config['eval'].get('eval_batch_size', config['training']['batch_size'])
    context_length = config['model']['context_length']

    # Evaluate on training set
    train_losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch_from_memmap(
            rngs, train_dataset, batch_size, context_length
        )

        logits = model(inputs)
        # Reshape for cross-entropy: (B, S, V) -> (B*S, V) and (B, S) -> (B*S,)
        B, S, V = logits.shape
        loss = model_module.cross_entropy_loss(
            logits.reshape(B * S, V),
            targets.reshape(B * S)
        )
        train_losses.append(loss.item())
    losses['train'] = np.mean(train_losses)

    # Evaluate on validation set if available
    if val_dataset is not None:
        val_losses = []
        for _ in range(eval_iters):
            inputs, targets = get_batch_from_memmap(
                rngs, val_dataset, batch_size, context_length
            )
            logits = model(inputs)
            B, S, V = logits.shape
            loss = model_module.cross_entropy_loss(
                logits.reshape(B * S, V),
                targets.reshape(B * S)
            )
            val_losses.append(loss.item())
        losses['val'] = np.mean(val_losses)

    return losses


def train(config_path: str, use_wandb: bool = False, run_name: Optional[str] = None):
    """
    Main training loop.

    Args:
        config_path: Path to YAML configuration file
        use_wandb: Whether to use Weights & Biases for logging
        run_name: Optional name for this training run
    """
    # Load configuration
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    # Setup logging
    wandb_logger = setup_logging(use_wandb, config, run_name)

    # Set random seed for reproducibility
    seed = config['training'].get('seed', 42)
    RNGS = nnx.Rngs(seed)
    np.random.seed(seed)

    # Load datasets with memory mapping
    print("\nLoading datasets...")
    train_dataset = MemMapDataset(config['data']['train_path'])

    val_dataset = None
    if 'val_path' in config['data'] and config['data']['val_path']:
        val_dataset = MemMapDataset(config['data']['val_path'])

    # Initialize model
    print("\nInitializing model...")
    model = model_module.TransformerLM(
        rngs=rngs,
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        theta=config['model']['theta'],
        vocab_size=config['model']['vocab_size'],
        context_length=config['model']['context_length'],
        num_layers=config['model']['num_layers']
    )

    # Count parameters
    total_params = sum(p.size for p in jax.tree.leaves(nnx.state(model)) if isinstance(p, Array))
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer
    optimizer_config = config['optimizer']
    if optimizer_config['type'].lower() == 'adamw':
        optimizer = model_module.AdamW(
            lr=optimizer_config['lr'],
            betas=tuple(optimizer_config['betas']),
            weight_decay=optimizer_config['weight_decay'],
            eps=optimizer_config['eps']
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

    # Load checkpoint if resuming
    start_iter = 0
    checkpoint_config = config.get('checkpoint', {})
    if checkpoint_config.get('resume_from'):
        print(f"\nResuming from checkpoint: {checkpoint_config['resume_from']}")
        start_iter = model_module.load_checkpoint(
            checkpoint_config['resume_from'],
            model,
            optimizer
        )
        print(f"Resumed from iteration {start_iter}")

    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training configuration
    training_config = config['training']
    batch_size = training_config['batch_size']
    context_length = config['model']['context_length']
    max_iters = training_config['max_iters']
    eval_interval = training_config.get('eval_interval', 500)
    log_interval = training_config.get('log_interval', 100)
    gradient_clip = training_config.get('gradient_clip', 1.0)

    # Learning rate schedule configuration
    lr_schedule_config = config.get('lr_schedule', {})
    use_lr_schedule = 'warmup_iters' in lr_schedule_config

    # Training loop
    print("\n" + "="*80)
    print("Starting training")
    print("="*80)

    model.train()
    start_time = time.time()

    for iter_num in range(start_iter, max_iters):
        # Update learning rate according to schedule
        if use_lr_schedule:
            lr = model_module.get_lr_schedule(
                iter_num,
                lr_schedule_config['max_learning_rate'],
                lr_schedule_config['min_learning_rate'],
                lr_schedule_config['warmup_iters'],
                lr_schedule_config['cosine_cycle_iters']
            )
            # Update learning rate in optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = optimizer_config['lr']

        # Sample batch
        inputs, targets = get_batch_from_memmap(
            RNGS, train_dataset, batch_size, context_length
        )

        B, S = targets.shape

        # Compute loss and gradients
        loss, grad_state = nnx.value_and_grad(lambda model: model_module.cross_entropy_loss(model(inputs).reshape(B * S, -1), targets.reshape(B * S)))(model)
        # Gradient clipping
        if gradient_clip > 0:
            grad_state = model_module.gradient_clipping(grad_state, gradient_clip)

        # Optimizer step
        optimizer.update(model, grad_state)

        # Logging
        if iter_num % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (iter_num - start_iter + 1) * batch_size * context_length / elapsed

            print(f"Iter {iter_num:6d} | Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:,.0f}")

            if wandb_logger:
                wandb_logger.log({
                    'train/loss': loss.item(),
                    'train/lr': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                    'iteration': iter_num
                })

        # Evaluation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            print("\n" + "-"*80)
            print(f"Evaluating at iteration {iter_num}...")

            eval_iters = config['eval'].get('eval_iters', 100)
            losses = estimate_loss(
                RNGS, model, train_dataset, val_dataset, config, eval_iters
            )

            print(f"Train Loss: {losses['train']:.4f}")
            if 'val' in losses:
                print(f"Val Loss:   {losses['val']:.4f}")
            print("-"*80 + "\n")

            if wandb_logger:
                log_dict = {
                    'eval/train_loss': losses['train'],
                    'iteration': iter_num
                }
                if 'val' in losses:
                    log_dict['eval/val_loss'] = losses['val']
                wandb_logger.log(log_dict)

        # Save checkpoint
        save_interval = checkpoint_config.get('save_interval', 5000)
        if iter_num > 0 and (iter_num % save_interval == 0 or iter_num == max_iters - 1):
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num}.pt"
            print(f"Saving checkpoint to {checkpoint_path}")
            model_module.save_checkpoint(model, optimizer, iter_num, checkpoint_path)

            # Also save a "latest" checkpoint
            latest_path = checkpoint_dir / "checkpoint_latest.pt"
            model_module.save_checkpoint(model, optimizer, iter_num, latest_path)

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

    # Final checkpoint
    final_path = checkpoint_dir / "checkpoint_final.pt"
    print(f"\nSaving final checkpoint to {final_path}")
    model_module.save_checkpoint(model, optimizer, max_iters - 1, final_path)

    if wandb_logger:
        wandb_logger.finish()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')
    parser.add_argument(
        '--config',
        type=str,
        default='cs336_jax_basics/config.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this training run (for W&B)'
    )

    args = parser.parse_args()

    train(args.config, use_wandb=args.wandb, run_name=args.run_name)


if __name__ == '__main__':
    main()
