"""
Flask web app for interacting with trained GPT model.
"""

from flask import Flask, render_template, request, jsonify
import torch
import pickle
from pathlib import Path
from models.transformers import GPTModel
from config.model_config import SmallModelConfig

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    global model, tokenizer, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_file = Path('tokenizer.pkl')
    if not tokenizer_file.exists():
        raise FileNotFoundError("Tokenizer not found. Train the model first using train_example.py")
    
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded")
    
    # Load model
    checkpoint_file = Path('model_checkpoint.pt')
    if not checkpoint_file.exists():
        raise FileNotFoundError("Model checkpoint not found. Train the model first using train_example.py")
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Handle both old format (direct state dict) and new format (dict with 'model_state_dict')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        vocab_size = checkpoint.get('vocab_size', 512)
        model_state = checkpoint['model_state_dict']
    else:
        # Old format: checkpoint is just the state dict
        vocab_size = 512
        model_state = checkpoint
    
    # Initialize model with config
    model_config = SmallModelConfig()
    model_config.vocab_size = vocab_size
    
    model = GPTModel(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        d_ff=model_config.d_ff,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout
    )
    
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    print("Model loaded and ready for inference")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate text based on the input prompt."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt.strip():
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        max_new_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.8)
        top_k = data.get('top_k', 10)
        
        # Encode prompt
        prompt_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode output
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        return jsonify({
            'success': True,
            'prompt': prompt,
            'generated': generated_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Check if model is loaded."""
    return jsonify({
        'status': 'ready' if model is not None else 'not_ready',
        'device': device
    })


if __name__ == '__main__':
    try:
        load_model_and_tokenizer()
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_example.py first to train the model.")
