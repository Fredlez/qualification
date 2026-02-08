"""
Training utilities for the language model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class TextDataset(Dataset):
    """Dataset for language modeling."""
    
    def __init__(self, token_ids: List[int], block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size
        
    def __len__(self):
        return max(0, len(self.token_ids) - self.block_size)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y


def create_dataloader(
    token_ids: List[int],
    block_size: int,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = TextDataset(token_ids, block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    return dataloader


class Trainer:
    """Trainer class for model training."""
    
    def __init__(self, model, train_dataloader, val_dataloader, learning_rate, device, max_iters, eval_interval, eval_iters):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def train(self):
        """Train the model and return loss history."""
        loss_history = []
        for iteration in range(self.max_iters):
            total_loss = 0
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            loss_history.append(total_loss)
            if iteration % self.eval_interval == 0:
                print(f"Iteration {iteration}/{self.max_iters}, Loss: {total_loss:.4f}")
        
        return loss_history


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y