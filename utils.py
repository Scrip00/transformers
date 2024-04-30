import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')

class TextDataset(Dataset):
    def __init__(self, texts, vocab, stemmer):
        self.vocab = vocab
        self.stemmer = stemmer
        self.pairs = [self.process_text(text) for text in texts]

    def process_text(self, text):
        tokens = word_tokenize(text.lower())
        indices = [self.vocab.get(self.stemmer.stem(token), self.vocab['<unk>']) for token in tokens]
        # Assuming next-word prediction; adjust as necessary for your specific task
        inputs = torch.tensor(indices[:-1], dtype=torch.long)
        targets = torch.tensor(indices[1:], dtype=torch.long)
        return inputs, targets

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_batch(batch, word_to_index):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=word_to_index['<pad>'])
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=word_to_index['<pad>'])
    return inputs_padded, targets_padded

def build_vocab(texts):
    stemmer = PorterStemmer()
    vocab_set = set()
    for text in texts:
        tokens = word_tokenize(text.lower())
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        vocab_set.update(stemmed_tokens)
    vocab = ['<sos>', '<eos>', '<unk>', '<pad>'] + sorted(list(vocab_set))
    return {word: index for index, word in enumerate(vocab)}

def load_wikipedia_subset(percentage=0.01, cache_dir="./cache"):
    dataset = load_dataset('wikipedia', '20220301.en', split='train', cache_dir=cache_dir)
    sample_size = int(len(dataset) * percentage)
    random.seed(42)
    sampled_indices = random.sample(range(len(dataset)), sample_size)
    sampled_texts = [dataset[i]['text'] for i in sampled_indices]
    return sampled_texts

def prepare_data_loaders(texts, config):
    vocab = build_vocab(texts)
    stemmer = PorterStemmer()
    dataset = TextDataset(texts, vocab, stemmer)
    total = len(dataset)
    train_size = int(config['split'][0] * total)
    val_size = int(config['split'][1] * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    collate_fn = lambda batch: collate_batch(batch, vocab)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab

def main():
    config = {
        'split': [0.8, 0.1, 0.1],
        'embed_dim': 512,
        'hidden_dim': 512,
        'epochs': 10,
        'learning_rate': 1e-3,
        'batch_size': 32,
    }
    
    wiki_texts = load_wikipedia_subset(0.01)
    train_loader, val_loader, test_loader, word_to_index = prepare_data_loaders(wiki_texts, config)

    print(f"Datasets loaded: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

if __name__ == "__main__":
    main()
