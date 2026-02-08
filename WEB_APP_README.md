# GPT Agent Web Interface

A simple web interface to interact with your trained GPT model.

## Features

- 🎯 **Easy Prompt Input**: Enter any prompt and generate text
- ⚙️ **Configurable Parameters**:
  - **Max Tokens**: Control the length of generated text (10-200)
  - **Temperature**: Control randomness (0.0-2.0)
    - Lower values = more deterministic
    - Higher values = more creative/random
  - **Top K**: Limit vocabulary to top K tokens for diversity
- 💾 **Model Persistence**: Uses your saved model checkpoint and tokenizer
- 🚀 **Real-time Generation**: Instant feedback in the browser

## Prerequisites

1. Train your model first using `train_example.py`:

   ```bash
   uv run python train_example.py
   ```

   This creates:
   - `model_checkpoint.pt` - Your trained model
   - `tokenizer.pkl` - Your trained tokenizer
   - `optimizer_checkpoint.pt` - Optimizer state for future training
   - `training_log.json` - Training history

2. Install Flask (if not already installed):
   ```bash
   pip install flask
   ```

## Running the Web App

```bash
uv run python app.py
```

The app will start on `http://localhost:5000`

## How to Use

1. Open your browser to `http://localhost:5000`
2. Enter a prompt (e.g., "To be, or not to be")
3. (Optional) Adjust generation parameters:
   - **Max Tokens**: How long the generated text should be
   - **Temperature**: How creative vs predictable the output should be
   - **Top K**: How many candidate tokens to consider
4. Click "Generate Text"
5. View your prompt and the model's generated continuation

## API Endpoints

### GET `/`

Returns the web interface

### POST `/generate`

Generates text based on a prompt

**Request:**

```json
{
  "prompt": "To be, or not to be",
  "max_tokens": 50,
  "temperature": 0.8,
  "top_k": 10
}
```

**Response:**

```json
{
  "success": true,
  "prompt": "To be, or not to be",
  "generated": "To be, or not to be, that is the question..."
}
```

### GET `/health`

Check if the model is loaded and ready

## Troubleshooting

**"Model checkpoint not found"**

- Run `train_example.py` first to train and save the model

**"Tokenizer not found"**

- Run `train_example.py` first to train the tokenizer

**Slow generation**

- If you don't have a GPU, CPU generation will be slower
- Reduce `max_tokens` for faster generation
- The first generation takes slightly longer due to model initialization

**CUDA out of memory**

- Reduce `max_tokens` value
- Close other GPU-intensive applications
- The app will automatically fall back to CPU if CUDA isn't available

## Model Information

The web app uses:

- **Model**: Small GPT-based transformer
- **Vocab Size**: 512 tokens (from your trained tokenizer)
- **Training Data**: WikiText-2 dataset (first 5%)
- **Checkpoint**: Incrementally trained across sessions

Each time you run `train_example.py`, it:

1. Loads the previous checkpoint
2. Trains for 100 more iterations
3. Saves the improved model back to the same checkpoint
4. Logs training history in `training_log.json`
