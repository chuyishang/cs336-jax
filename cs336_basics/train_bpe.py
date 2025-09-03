import os
import regex as re

from collections import Counter

def _process_chunk(text: str, pattern: str) -> Counter[tuple[bytes]]:
    """Helper function for multiprocessing in load_and_pretokenize."""
    pretoken_counts = Counter()
    for m in re.finditer(pattern, text):
        chunk = m.group(0)
        encoded_bytes = chunk.encode('utf-8')
        byte_array = tuple([bytes([b]) for b in encoded_bytes])
        pretoken_counts[byte_array] += 1
    return pretoken_counts

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
        from multiprocessing import Pool, cpu_count
        from functools import partial

        # Read the corpus
        with open(input_path, "r") as f:
            corpus = f.read()
        
        # Prepare the regex pattern
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        special_tokens_pat = "|".join([re.escape(s) for s in special_tokens])
        
        # Split corpus by special tokens
        corpus_parts = re.split(special_tokens_pat, corpus)
        
        # Only use multiprocessing for larger corpora with multiple parts
        if len(corpus_parts) > 1 and len(corpus) > 50000:  # Use multiprocessing for larger files
            num_processes = min(cpu_count(), len(corpus_parts))
            with Pool(num_processes) as pool:
                chunk_counters = pool.map(partial(_process_chunk, pattern=PAT), corpus_parts)
        else:
            # Single-threaded for small files to avoid multiprocessing overhead
            chunk_counters = [_process_chunk(part, PAT) for part in corpus_parts]
        
        # Combine all counters
        final_counts = Counter()
        for counter in chunk_counters:
            final_counts.update(counter)
        
        return final_counts

    def count_pairs(pretokens: Counter[tuple[bytes]]) -> Counter[tuple[bytes, bytes]]:
        """Count all adjacent byte pairs in the pretokens."""
        bp_counts = Counter()
        for ptk_bytes in pretokens.keys():
            for i in range(len(ptk_bytes)-1):
                bp_counts[(ptk_bytes[i], ptk_bytes[i+1])] += pretokens[ptk_bytes]
        return bp_counts

    def get_most_frequent_pair(bp_counts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
        """Get the most frequent pair from the pair counts."""
        most_frequent_pair, _ = max(bp_counts.items(), key=lambda x: (x[1], x[0]))
        return most_frequent_pair
    
    def update_merges_and_pairs(pretokens: Counter[tuple[bytes]], 
                               bp_counts: Counter[tuple[bytes, bytes]], 
                               pair: tuple[bytes, bytes]) -> tuple[Counter[tuple[bytes]], Counter[tuple[bytes, bytes]]]:
        """Update pretokens and incrementally update pair counts."""
        new_pretokens = Counter()
        
        for pretoken_bytes, count in pretokens.items():
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
    bp_counts = count_pairs(pretoken_counts)
    
    # Optimized training loop with incremental updates
    while len(vocab) < vocab_size:
        most_frequent_pair = get_most_frequent_pair(bp_counts)
        # Add new token to vocab
        vocab[len(vocab)] = most_frequent_pair[0] + most_frequent_pair[1] # add to merge tokens
        merges.append((most_frequent_pair[0], most_frequent_pair[1]))
        
        # Incrementally update pretokens and pair counts
        pretoken_counts, bp_counts = update_merges_and_pairs(pretoken_counts, bp_counts, most_frequent_pair)
    
    return vocab, merges
