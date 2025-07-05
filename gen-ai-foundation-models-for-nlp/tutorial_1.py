import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import optim

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the text classification model
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

def collate_batch(batch, tokenizer, vocab):
    label_list, text_list, offsets = [], [], [0]
    
    for (_label, _text) in batch:
        label_list.append(_label - 1)  # Convert to 0-based indexing
        processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    
    return label_list, text_list, offsets

def main():
    # Hyperparameters
    BATCH_SIZE = 3
    EMBED_DIM = 64
    NUM_CLASS = 4  # AG_NEWS has 4 classes
    
    # Load dataset
    train_iter = AG_NEWS(split='train')
    tokenizer = get_tokenizer('basic_english')
    
    # Build vocabulary
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenizer),
        specials=["<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    
    # Create data loaders
    train_iter = AG_NEWS(split='train')
    train_loader = DataLoader(
        train_iter,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, vocab)
    )
    
    # Initialize model
    vocab_size = len(vocab)
    model = TextClassificationModel(vocab_size, EMBED_DIM, NUM_CLASS)
    
    # Sample prediction
    def predict(text, model, vocab, tokenizer):
        with torch.no_grad():
            text = torch.tensor(vocab(tokenizer(text)))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1  # Convert back to 1-based for display
    
    # Example usage
    sample_text = "This is a sample news article about technology"
    predicted_label = predict(sample_text, model, vocab, tokenizer)
    print(f"Sample text: '{sample_text}'")
    print(f"Predicted label: {predicted_label}")
    
    # Print model architecture
    print("\nModel architecture:")
    print(model)
    
    # Print a batch of data
    print("\nSample batch from DataLoader:")
    labels, text, offsets = next(iter(train_loader))
    print(f"Labels (0-based): {labels}")
    print(f"Text indices: {text}")
    print(f"Offsets: {offsets}")
    
    # Show model output for the batch
    with torch.no_grad():
        output = model(text, offsets)
        print("\nModel output (logits) for the batch:")
        print(output)
        print("Predicted classes (1-based):", output.argmax(1).numpy() + 1)

if __name__ == "__main__":
    main()
