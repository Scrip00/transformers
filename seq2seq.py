from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from collections import Counter

config = {
    'split': [0.8, 0.1, 0.1],
    'min_freq': 2
}

def tokenizer(text):
    return [token for token in word_tokenize(text)]

def build_dataset(path, config, stemmer=PorterStemmer()):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = [word_tokenize(line.strip()) for line in lines]
    data = [[stemmer.stem(token) for token in seq] for seq in data]
    data = [['<sos>'] + s + ['<eos>'] for s in data]

    train, test_val = train_test_split(data, train_size=config['split'][0], random_state=777)
    test, val = train_test_split(test_val, train_size=config['split'][1] / (config['split'][1] + config['split'][2]), random_state=777)
    token_counts = Counter()
    for sequence in train:
        token_counts.update(sequence)
    
    vocab = [token for token, count in token_counts.items() if count >= config['min_freq']]
    return train, test, val, vocab

train, test, val, vocab = build_dataset('./data/dialogs.txt', config)

print(train)
