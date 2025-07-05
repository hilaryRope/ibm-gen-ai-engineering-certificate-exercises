# PyTorch Text Classification Project

This project implements document categorization using the AG_NEWS dataset with PyTorch and Torchtext. The model uses an embedding bag layer followed by a fully connected layer for text classification.

## Features

- Text preprocessing and tokenization
- Vocabulary building and numerical encoding
- Custom data collation for batching
- Neural network model with embedding bag
- Example prediction pipeline
- Batch processing support

## Model Architecture

The model consists of:
1. `EmbeddingBag` layer for efficient text embedding
2. Linear layer for classification
3. Custom weight initialization

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install torch torchtext
```

## Usage

1. Run the training and evaluation script:
```bash
python tutorial_1.py
```

2. The script will:
   - Load and preprocess the AG_NEWS dataset
   - Build vocabulary from the training data
   - Initialize the text classification model
   - Show a sample prediction
   - Display model architecture
   - Process and show a sample batch of data

## Output

The script provides the following output:
- Sample text prediction
- Model architecture summary
- Sample batch details (labels, text indices, offsets)
- Model output (logits) for the batch
- Predicted classes for the batch

## File Structure

- `tutorial_1.py`: Main script containing the implementation
- `README.md`: This documentation file

## Requirements

- Python 3.7+
- PyTorch
- Torchtext
