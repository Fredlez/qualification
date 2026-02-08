"""
Configuration file for model hyperparameters and training settings.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for GPT model architecture."""
    vocab_size: int = 50257  # Word ammount
    d_model: int = 768       # Embedding dimension
    num_heads: int = 12      # Number of attention heads
    num_layers: int = 12     # Number of transformer blocks
    d_ff: int = 3072         # Feed-forward dimension
    max_seq_len: int = 1024  # Maximum sequence length
    dropout: float = 0.1     # Dropout rate


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 8          # Batch size
    block_size: int = 256        # Context length
    learning_rate: float = 3e-4  # Learning rate
    max_iters: int = 100         # Maximum training iterations
    eval_interval: int = 1       # Evaluation frequency
    eval_iters: int = 5          # How many times to evalueate iterations during training
    device: str = "cuda"         # Trainis on GPU
    

# Small model for testing
@dataclass
class SmallModelConfig:
    """Smaller model for quick testing."""
    vocab_size: int = 512
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1


# Medium model configuration
@dataclass
class MediumModelConfig:
    """Medium-sized model."""
    vocab_size: int = 50257
    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    d_ff: int = 4096
    max_seq_len: int = 1024
    dropout: float = 0.1