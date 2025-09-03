import pickle
import regex as re
import multiprocessing as mp

from functools import partial
from typing import Any, IO, BinaryIO, Iterable


def init_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def _process_chunk(pattern: str, text: str, special_tokens: list[str] | None) -> list[list[bytes]]:
    if special_tokens and text in special_tokens:
        return [[text.encode('utf-8')]]
    result = []
    for m in re.finditer(pattern, text):
        chunk = m.group(0)
        encoded_bytes = chunk.encode('utf-8')
        result.append([bytes([b]) for b in encoded_bytes])
    return result


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod 
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    
    def pretokenize_text(self, text: str) -> list[list[bytes]]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            # Sort by length (longest first) to handle overlapping tokens
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_tokens_pat = "|".join([re.escape(s) for s in sorted_special_tokens])
            text_parts = re.split(f"({special_tokens_pat})", text)
        else:
            text_parts = [text]
        pretokens = []
        for part in text_parts:
            if not part:
                continue
            else:
                pretokens.extend(_process_chunk(pattern=PAT, text=part, special_tokens=self.special_tokens))
        return pretokens
    
    
    def merge_single_pretoken(self, pretoken: list[bytes]) -> list[bytes]:  
        for merge in self.merges:
            new_pretoken = []
            i = 0
            while i < len(pretoken): 
                if i < len(pretoken) - 1 and (pretoken[i], pretoken[i+1]) == merge:
                    new_pretoken.append(pretoken[i] + pretoken[i+1])
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            pretoken = new_pretoken
            if len(pretoken) == 1:
                break
        return pretoken
        

    def merge_pretokens(self, pretokens: list[list[bytes]]) -> list[bytes]:
        final_list = []
        for pretoken in pretokens:
            final_list.extend(self.merge_single_pretoken(pretoken))
        return final_list
    
    
    def convert_to_token_ids(self, token_list: list[bytes]) -> list[int]:
        token_ids = []
        for token in token_list:
            for token_id, token_bytes in self.vocab.items():
                if token_bytes == token:
                    token_ids.append(token_id)
                    break
        return token_ids


    def encode(self, text: str) -> list[int]:
        pretokens = self.pretokenize_text(text)
        token_list = self.merge_pretokens(pretokens)
        token_ids = self.convert_to_token_ids(token_list)
        return token_ids


    def decode(self, ids: list[int]) -> str:
        output = b"".join(self.vocab[id] for id in ids)
        return output.decode('utf-8', errors='replace')

    def encode_iterable(self, iterable: Iterable[str]):
        """Encode an iterable of strings, yielding token IDs one at a time."""
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id