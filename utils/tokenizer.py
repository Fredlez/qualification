"""
Simple tokenizer implementation for LLM training.
Based on Byte Pair Encoding (BPE) principles.
"""
from collections import Counter
from typing import List, Dict, Tuple
import pickle


class SimpleTokenizer:
    """Basic tokenizer using character-level encoding as a starting point."""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}  # Change to dict for O(1) lookup
        self.next_id = 256
        
    def get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def train(self, text: str, verbose: bool = False):
        # Initialize vocabulary with characters
        vocab = {i: chr(i) for i in range(256)}
        ids = list(text.encode('utf-8'))
        
        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = self.next_id
            self.next_id += 1
            ids = self.merge(ids, pair, new_id)
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            self.merges[pair] = new_id  # Store as dict for O(1) lookup
            
            if verbose and i % 100 == 0:
                print(f"Merge {i}/{num_merges}: {pair} -> {new_id}")
        
        self.vocab = vocab
    
    def encode(self, text: str) -> List[int]:
        ids = list(text.encode('utf-8'))
        while len(ids) > 1:
            stats = self.get_stats(ids)
            if not stats:
                break
            # Find the pair with the lowest merge ID
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            ids = self.merge(ids, pair, new_id)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        tokens = b"".join(
            self.vocab[idx].encode('latin-1') if isinstance(self.vocab[idx], str) else self.vocab[idx] 
            for idx in ids
        )
        return tokens.decode('utf-8', errors='replace')
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'vocab': self.vocab, 'merges': self.merges}, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.next_id = max(self.vocab.keys()) + 1


class GPTTokenizer:
    """GPT-style tokenizer."""
    pass