import os
import regex as re

from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback no-op progress bar
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else None
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])

        def update(self, n=1):
            pass

        def set_postfix(self, **kwargs):
            pass

def _process_chunk(text: str, pattern: str) -> Counter[tuple[bytes]]:
    """Helper function for multiprocessing in load_and_pretokenize."""
    pretoken_counts = Counter()
    for m in re.finditer(pattern, text):
        chunk = m.group(0)
        encoded_bytes = chunk.encode('utf-8')
        byte_array = tuple([bytes([b]) for b in encoded_bytes])
        pretoken_counts[byte_array] += 1
    return pretoken_counts

def _count_pairs_chunk(pretokens_items: list[tuple[tuple[bytes], int]]) -> Counter[tuple[bytes, bytes]]:
    """Helper function for parallel pair counting."""
    bp_counts = Counter()
    for ptk_bytes, count in pretokens_items:
        for i in range(len(ptk_bytes)-1):
            bp_counts[(ptk_bytes[i], ptk_bytes[i+1])] += count
    return bp_counts

def _update_pretoken(item: tuple[tuple[bytes], int], pair: tuple[bytes, bytes]) -> tuple[tuple[bytes], int, list[tuple[tuple[bytes, bytes], int]], list[tuple[tuple[bytes, bytes], int]]]:
    """Helper function for parallel pretoken updating.

    Returns:
        (new_pretoken_bytes, count, old_pairs_to_decrement, new_pairs_to_increment)
    """
    pretoken_bytes, count = item

    # Check if this pretoken contains the pair to be merged
    contains_pair = False
    for i in range(len(pretoken_bytes) - 1):
        if (pretoken_bytes[i], pretoken_bytes[i+1]) == pair:
            contains_pair = True
            break

    # If doesn't contain pair, return unchanged
    if not contains_pair:
        return (pretoken_bytes, count, [], [])

    # Collect old pairs to decrement
    old_pairs = []
    for i in range(len(pretoken_bytes) - 1):
        old_pair = (pretoken_bytes[i], pretoken_bytes[i+1])
        old_pairs.append((old_pair, count))

    # Merge the pair in this pretoken
    tokens = []
    i = 0
    while i < len(pretoken_bytes):
        if i < len(pretoken_bytes) - 1 and (pretoken_bytes[i], pretoken_bytes[i+1]) == pair:
            tokens.append(pair[0] + pair[1])
            i += 2
        else:
            tokens.append(pretoken_bytes[i])
            i += 1

    new_token_seq = tuple(tokens)

    # Collect new pairs to increment
    new_pairs = []
    for i in range(len(new_token_seq) - 1):
        new_pair = (new_token_seq[i], new_token_seq[i+1])
        new_pairs.append((new_pair, count))

    return (new_token_seq, count, old_pairs, new_pairs)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.

    Notes:
        - We first initialize the vocabulary. Initial vocab size = 256 since we are training a byte-level tokenizer.
        - Then we pre-tokenize the input dataset. In this step, we:
            - Use the regex pattern to split the input corpus into tokens
            - Represent this as dict[tuple[bytes], int]
            - Sum all byte-pair counts
        - Building blocks:
            - Symbol ((1,)): single byte
            - Word (tuple[bytes]): a sequence of bytes
    """
    
    def load_and_pretokenize(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter[tuple[bytes]]:
        """
        Loads a corpus from input path and pretokenizes it.

        Args:
            input_path: Path to corpus to load and pretokenize.
            special_tokens: List of special tokens to handle during tokenization.

        Returns:
            Counter object corresponding to each pretoken and their counts.
            Each key is a tuple of bytes e.g. (b't', b'e', b's', b't')
        """
        # Read the corpus
        with open(input_path, "r") as f:
            corpus = f.read()

        # Prepare the regex pattern
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Split corpus by special tokens to get document boundaries
        # This is particularly important for <|endoftext|> which delimits documents
        special_tokens_pat = "|".join([re.escape(s) for s in special_tokens])

        if special_tokens_pat:
            corpus_parts = re.split(special_tokens_pat, corpus)
        else:
            corpus_parts = [corpus]

        # Filter out empty parts
        corpus_parts = [part for part in corpus_parts if part.strip()]

        # Optimize chunking: Use document boundaries (split by special tokens) as natural chunks
        # This is more efficient than arbitrary splits and respects document structure
        chunks = []

        # Target chunk size for good load balancing
        # Aim for 4x the number of CPU cores to ensure good work distribution
        num_target_chunks = cpu_count() * 4

        if len(corpus_parts) >= num_target_chunks:
            # If we have enough documents, use them directly as chunks
            chunks = corpus_parts
        else:
            # If we have large documents, split them into smaller chunks
            # But try to keep document boundaries where possible
            target_chunk_size = max(5000, len(corpus) // num_target_chunks)

            for part in corpus_parts:
                if len(part) > target_chunk_size * 2:
                    # Split large documents into smaller chunks for better parallelization
                    for i in range(0, len(part), target_chunk_size):
                        chunk = part[i:i+target_chunk_size]
                        if chunk.strip():  # Only add non-empty chunks
                            chunks.append(chunk)
                else:
                    # Keep smaller documents as-is
                    chunks.append(part)

        # Use multiprocessing for larger corpora
        if len(chunks) > 1 and len(corpus) > 5000:  # Lower threshold for better performance
            num_processes = min(cpu_count(), len(chunks))
            with Pool(num_processes) as pool:
                with tqdm(total=len(chunks), desc="Pretokenizing", unit="doc", disable=not TQDM_AVAILABLE) as pbar:
                    chunk_counters = []
                    # Use imap for incremental progress updates
                    for result in pool.imap(partial(_process_chunk, pattern=PAT), chunks, chunksize=1):
                        chunk_counters.append(result)
                        pbar.update(1)
        else:
            # Single-threaded for small files to avoid multiprocessing overhead
            chunk_counters = []
            for chunk in tqdm(chunks, desc="Pretokenizing", unit="doc", disable=not TQDM_AVAILABLE):
                chunk_counters.append(_process_chunk(chunk, PAT))

        # Combine all counters efficiently
        final_counts = Counter()
        if len(chunk_counters) > 100:
            # Show progress for large number of chunks
            for counter in tqdm(chunk_counters, desc="Combining results", unit="doc", disable=not TQDM_AVAILABLE):
                final_counts.update(counter)
        else:
            # Don't show progress bar for small number of chunks
            for counter in chunk_counters:
                final_counts.update(counter)

        return final_counts

    def count_pairs(pretokens: Counter[tuple[bytes]], use_parallel: bool = True) -> Counter[tuple[bytes, bytes]]:
        """Count all adjacent byte pairs in the pretokens."""
        pretoken_items = list(pretokens.items())

        # Use parallel processing for large pretoken counts
        if use_parallel and len(pretoken_items) > 1000:
            # Split items into chunks for parallel processing
            num_processes = min(cpu_count(), max(1, len(pretoken_items) // 100))
            chunk_size = max(1, len(pretoken_items) // num_processes)
            chunks = [pretoken_items[i:i+chunk_size] for i in range(0, len(pretoken_items), chunk_size)]

            with Pool(num_processes) as pool:
                chunk_counts = pool.map(_count_pairs_chunk, chunks)

            # Combine all counters
            bp_counts = Counter()
            for counter in chunk_counts:
                bp_counts.update(counter)
        else:
            # Sequential processing for small datasets
            bp_counts = Counter()
            for ptk_bytes, count in pretoken_items:
                for i in range(len(ptk_bytes)-1):
                    bp_counts[(ptk_bytes[i], ptk_bytes[i+1])] += count

        return bp_counts

    def get_most_frequent_pair(bp_counts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
        """Get the most frequent pair from the pair counts."""
        most_frequent_pair, _ = max(bp_counts.items(), key=lambda x: (x[1], x[0]))
        return most_frequent_pair
    
    def update_merges_and_pairs(pretokens: Counter[tuple[bytes]],
                               bp_counts: Counter[tuple[bytes, bytes]],
                               pair: tuple[bytes, bytes],
                               use_parallel: bool = True) -> tuple[Counter[tuple[bytes]], Counter[tuple[bytes, bytes]]]:
        """Update pretokens and incrementally update pair counts."""
        pretoken_items = list(pretokens.items())

        # Use parallel processing for large pretoken sets
        if use_parallel and len(pretoken_items) > 1000:
            # Split items into chunks for parallel processing
            num_processes = min(cpu_count(), max(1, len(pretoken_items) // 100))
            chunk_size = max(1, len(pretoken_items) // num_processes)
            chunks = [pretoken_items[i:i+chunk_size] for i in range(0, len(pretoken_items), chunk_size)]

            with Pool(num_processes) as pool:
                results = pool.map(partial(_update_pretoken, pair=pair), [item for chunk in chunks for item in chunk])

            # Process results
            new_pretokens = Counter()
            pairs_to_decrement = Counter()
            pairs_to_increment = Counter()

            for new_token_seq, count, old_pairs, new_pairs in results:
                new_pretokens[new_token_seq] = count
                for old_pair, pair_count in old_pairs:
                    pairs_to_decrement[old_pair] += pair_count
                for new_pair, pair_count in new_pairs:
                    pairs_to_increment[new_pair] += pair_count

            # Update bp_counts
            for old_pair, decrement in pairs_to_decrement.items():
                bp_counts[old_pair] -= decrement
                if bp_counts[old_pair] <= 0:
                    del bp_counts[old_pair]

            for new_pair, increment in pairs_to_increment.items():
                bp_counts[new_pair] += increment

        else:
            # Sequential processing for small datasets
            new_pretokens = Counter()

            for pretoken_bytes, count in pretoken_items:
                # Check if this pretoken contains the pair to be merged
                contains_pair = False
                for i in range(len(pretoken_bytes) - 1):
                    if (pretoken_bytes[i], pretoken_bytes[i+1]) == pair:
                        contains_pair = True
                        break

                # If doesn't contain pair, copy to new dict
                if not contains_pair:
                    new_pretokens[pretoken_bytes] = count
                    continue

                # If contains pair, decrement pair counts by contribution of this pretoken
                for i in range(len(pretoken_bytes) - 1):
                    old_pair = (pretoken_bytes[i], pretoken_bytes[i+1])
                    bp_counts[old_pair] -= count
                    if bp_counts[old_pair] <= 0:
                        del bp_counts[old_pair]

                # Merge the pair in this pretoken
                tokens = []
                i = 0
                while i < len(pretoken_bytes):
                    if i < len(pretoken_bytes) - 1 and (pretoken_bytes[i], pretoken_bytes[i+1]) == pair:
                        tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        tokens.append(pretoken_bytes[i])
                        i += 1

                new_token_seq = tuple(tokens)
                new_pretokens[new_token_seq] = count

                # Increment new pair counts for the merged sequence
                for i in range(len(new_token_seq) - 1):
                    new_pair = (new_token_seq[i], new_token_seq[i+1])
                    bp_counts[new_pair] += count

        return new_pretokens, bp_counts
    
    
    # Vocab initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = bytes(special_token, 'utf-8')
    merges = []

    # Load and pretokenize
    pretoken_counts = load_and_pretokenize(input_path, special_tokens)

    # Initialize pair counts once
    print("Counting initial byte pairs...")
    bp_counts = count_pairs(pretoken_counts)

    # Calculate total merges needed
    initial_vocab_size = len(vocab)
    num_merges = vocab_size - initial_vocab_size

    # Optimized training loop with incremental updates
    with tqdm(total=num_merges, desc="Training BPE", unit="merge", disable=not TQDM_AVAILABLE) as pbar:
        while len(vocab) < vocab_size:
            most_frequent_pair = get_most_frequent_pair(bp_counts)
            # Add new token to vocab
            vocab[len(vocab)] = most_frequent_pair[0] + most_frequent_pair[1] # add to merge tokens
            merges.append((most_frequent_pair[0], most_frequent_pair[1]))

            # Update progress bar with info about current merge
            try:
                pair_str = f"{most_frequent_pair[0][:10]}+{most_frequent_pair[1][:10]}"
            except:
                pair_str = "binary"
            pbar.set_postfix({"vocab_size": len(vocab), "last_merge": pair_str})

            # Incrementally update pretokens and pair counts
            pretoken_counts, bp_counts = update_merges_and_pairs(pretoken_counts, bp_counts, most_frequent_pair)

            pbar.update(1)

    return vocab, merges
