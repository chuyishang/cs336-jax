"""
Prepare training data by training a BPE tokenizer and converting text files to binary format.

This script:
1. Trains a BPE tokenizer on the training data
2. Tokenizes both train and validation sets
3. Saves tokenized data as binary files (np.uint16)
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


def train_tokenizer(
    train_path: str,
    vocab_size: int = 10000,
    special_tokens: list[str] | None = None,
    output_dir: str = "tokenizer"
):
    """
    Train a BPE tokenizer on the training data.

    Args:
        train_path: Path to training text file
        vocab_size: Size of vocabulary to train
        special_tokens: List of special tokens
        output_dir: Directory to save tokenizer files

    Returns:
        Tokenizer instance
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    print(f"\n{'='*80}")
    print("Training BPE Tokenizer")
    print(f"{'='*80}")
    print(f"Training data: {train_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    # Train BPE
    print("\nTraining BPE (this may take a while for large files)...")
    vocab, merges = train_bpe(
        input_path=train_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    print(f"✓ Training complete!")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Number of merges: {len(merges)}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save vocab and merges
    vocab_file = output_path / "vocab.pkl"
    merges_file = output_path / "merges.pkl"

    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    with open(merges_file, 'wb') as f:
        pickle.dump(merges, f)

    print(f"\n✓ Tokenizer saved to {output_dir}/")
    print(f"  - vocab.pkl")
    print(f"  - merges.pkl")

    # Create tokenizer instance
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer


def tokenize_file(
    input_path: str,
    output_path: str,
    tokenizer: Tokenizer,
    chunk_size: int = 1000000,
):
    """
    Tokenize a text file and save as binary.

    Args:
        input_path: Path to input text file
        output_path: Path to output binary file
        tokenizer: Trained tokenizer
        chunk_size: Number of characters to process at once
    """
    print(f"\nTokenizing {input_path}...")

    # Read file size
    file_size = Path(input_path).stat().st_size
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")

    all_token_ids = []

    # Read and tokenize in chunks to handle large files
    with open(input_path, 'r', encoding='utf-8') as f:
        chunk_num = 0
        while True:
            text = f.read(chunk_size)
            if not text:
                break

            chunk_num += 1
            token_ids = tokenizer.encode(text)
            all_token_ids.extend(token_ids)

            if chunk_num % 10 == 0:
                print(f"  Processed {chunk_num} chunks, {len(all_token_ids):,} tokens so far...")

    print(f"✓ Tokenization complete!")
    print(f"  Total tokens: {len(all_token_ids):,}")

    # Convert to numpy array and save as binary
    token_array = np.array(all_token_ids, dtype=np.uint16)

    # Check if any token ID exceeds uint16 max
    max_token_id = max(all_token_ids)
    if max_token_id >= 65536:
        print(f"Warning: Max token ID ({max_token_id}) exceeds uint16 range!")
        print("Using uint32 instead...")
        token_array = np.array(all_token_ids, dtype=np.uint32)

    token_array.tofile(output_path)

    output_size = Path(output_path).stat().st_size
    print(f"✓ Saved to {output_path}")
    print(f"  Output size: {output_size / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {file_size / output_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data by tokenizing text files')
    parser.add_argument(
        '--train-path',
        type=str,
        required=True,
        help='Path to training text file'
    )
    parser.add_argument(
        '--val-path',
        type=str,
        default=None,
        help='Path to validation text file (optional)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=10000,
        help='Vocabulary size for BPE tokenizer (default: 10000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Directory to save tokenized files (default: data)'
    )
    parser.add_argument(
        '--tokenizer-dir',
        type=str,
        default='tokenizer',
        help='Directory to save/load tokenizer (default: tokenizer)'
    )
    parser.add_argument(
        '--special-tokens',
        nargs='+',
        default=['<|endoftext|>'],
        help='Special tokens to add (default: <|endoftext|>)'
    )
    parser.add_argument(
        '--use-existing-tokenizer',
        action='store_true',
        help='Use existing tokenizer from tokenizer-dir instead of training new one'
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Data Preparation Pipeline")
    print(f"{'='*80}")

    # Train or load tokenizer
    if args.use_existing_tokenizer:
        print(f"\nLoading existing tokenizer from {args.tokenizer_dir}/")
        tokenizer = Tokenizer.from_files(
            vocab_filepath=f"{args.tokenizer_dir}/vocab.pkl",
            merges_filepath=f"{args.tokenizer_dir}/merges.pkl",
            special_tokens=args.special_tokens
        )
        print("✓ Tokenizer loaded")
    else:
        tokenizer = train_tokenizer(
            train_path=args.train_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            output_dir=args.tokenizer_dir
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize training data
    print(f"\n{'='*80}")
    print("Tokenizing Training Data")
    print(f"{'='*80}")
    train_output = output_dir / "train.bin"
    tokenize_file(args.train_path, train_output, tokenizer)

    # Tokenize validation data if provided
    if args.val_path:
        print(f"\n{'='*80}")
        print("Tokenizing Validation Data")
        print(f"{'='*80}")
        val_output = output_dir / "val.bin"
        tokenize_file(args.val_path, val_output, tokenizer)

    print(f"\n{'='*80}")
    print("Data Preparation Complete!")
    print(f"{'='*80}")
    print(f"\nTokenized files saved to {args.output_dir}/:")
    print(f"  - train.bin")
    if args.val_path:
        print(f"  - val.bin")
    print(f"\nTokenizer saved to {args.tokenizer_dir}/:")
    print(f"  - vocab.pkl")
    print(f"  - merges.pkl")
    print(f"\nYou can now use these files for training!")
    print(f"Update your config.yaml to use:")
    print(f"  train_path: train.bin")
    if args.val_path:
        print(f"  val_path: val.bin")


if __name__ == '__main__':
    main()
