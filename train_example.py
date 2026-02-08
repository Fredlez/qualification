"""
Example training script for the GPT model.
This demonstrates how to train the model on a simple text dataset.
"""

import torch
import sys
import json
import pickle
from pathlib import Path
sys.path.append('.')

from models.transformers import GPTModel
from utils.tokenizer import SimpleTokenizer
from utils.training import create_dataloader, Trainer
from config.model_config import SmallModelConfig, TrainingConfig
from datasets import load_dataset


def load_training_log():
    """Load existing training log or create new one."""
    log_file = Path('training_log.json')
    if log_file.exists():
        with open(log_file, 'r') as f:
            return json.load(f)
    return {"total_iterations": 0, "sessions": []}


def save_training_log(log_data):
    """Save training log to file."""
    with open('training_log.json', 'w') as f:
        json.dump(log_data, f, indent=2)


def main():
    # Load training log
    training_log = load_training_log()
    current_session = {
        "session_number": len(training_log["sessions"]) + 1,
        "iterations": [],
        "final_loss": None
    }
    
    print("Step 1: Loading training data from HuggingFace...")
    # Load a small dataset for training
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train[:5%]")
    sample_text = "\n".join(dataset["text"][:1000])
    
    print("Step 2: Loading or training tokenizer...")
    tokenizer_file = Path('tokenizer.pkl')
    if tokenizer_file.exists():
        # Load the currently existing tokenizer
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Loaded existing tokenizer")
    else:
        # Train a new tokenizer with the sample text
        tokenizer = SimpleTokenizer(vocab_size=512)
        tokenizer.train(sample_text, verbose=True)
        # Save tokenizer for future training
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)
        print("Tokenizer trained and saved")
    
    print("\nStep 3: Encoding text...")
    token_ids = tokenizer.encode(sample_text)
    print(f"Total tokens: {len(token_ids)}")
    
    # Split it into training and validations
    split_idx = int(0.9 * len(token_ids))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    
    print("\nStep 4: Creating dataloaders...")
    # Creates dataloaders
    train_config = TrainingConfig()
    train_dataloader = create_dataloader(
        train_ids,
        block_size=train_config.block_size,
        batch_size=train_config.batch_size
    )
    val_dataloader = create_dataloader(
        val_ids,
        block_size=train_config.block_size,
        batch_size=train_config.batch_size,
        shuffle=False
    )
    
    print("\nStep 5: Initializing model...")
    # Initialize the model with its config
    model_config = SmallModelConfig()
    model_config.vocab_size = len(tokenizer.vocab)
    model_config.max_seq_len = train_config.block_size
    
    model = GPTModel(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        d_ff=model_config.d_ff,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Load the most recent checkpoint if it exists so it can continue training
    checkpoint_file = Path('model_checkpoint.pt')
    optimizer_file = Path('optimizer_checkpoint.pt')
    
    try:
        checkpoint = torch.load(checkpoint_file)
        # Handle both old format (direct state dict) and new format (dict with 'model_state_dict')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Old format: checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        print("Loaded model checkpoint. Continuing training...")
    except FileNotFoundError:
        print("No checkpoint found. Training from scratch...")
    
    print("\nStep 6: Training model...")
    # Initialize trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=train_config.learning_rate,
        device=device,
        max_iters=train_config.max_iters,
        eval_interval=train_config.eval_interval,
        eval_iters=train_config.eval_iters
    )
    
    # Load the optimizer state if it is available so the training can continue
    try:
        if optimizer_file.exists():
            optimizer_checkpoint = torch.load(optimizer_file)
            trainer.optimizer.load_state_dict(optimizer_checkpoint)
            print("Loaded optimizer state")
    except:
        pass
    
    # Train the model
    loss_history = trainer.train()
    current_session["iterations"] = loss_history
    current_session["final_loss"] = loss_history[-1] if loss_history else None
    
    print("\nStep 7: Saving model, optimizer, and training data...")
    # Save the checkpoint at the end of training so it can be loaded in future training sessions
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model_config.vocab_size,
        'config': model_config.__dict__
    }, 'model_checkpoint.pt')
    
    # Save optimizer state so it can continue seamlessly
    torch.save(trainer.optimizer.state_dict(), 'optimizer_checkpoint.pt')
    
    # Save the current tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Update and save training log
    training_log["total_iterations"] += 100
    training_log["sessions"].append(current_session)
    save_training_log(training_log)
    print(f"Training log saved. Total iterations so far: {training_log['total_iterations']}")
    
    print("\nStep 8: Generating text...")
    # Generate some text
    model.eval()
    prompt = "Olegg had a dream about the future of AI. In his vision, he saw a world where "
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=10
        )
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    print("\nTraining complete! Model, optimizer, and tokenizer saved.")
    print(f"Session {current_session['session_number']} completed with final loss: {current_session['final_loss']:.4f}")
    
    # Reload the model from the checkpoint to make sure it works
    final_checkpoint = torch.load('model_checkpoint.pt')
    if isinstance(final_checkpoint, dict) and 'model_state_dict' in final_checkpoint:
        model.load_state_dict(final_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(final_checkpoint)


if __name__ == "__main__":
    main()
