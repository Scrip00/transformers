import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')

def build_dataset(config):
    data_path = './data/dialogs.txt'

    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    input_seqs = []
    target_seqs = []
    for line in lines:
        input_seq, target_seq = line.strip().split('\t')
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)

    stemmer = PorterStemmer()

    vocab_set = set()
    for seq in input_seqs + target_seqs:
        tokens = word_tokenize(seq.lower())  # Tokenization and lowercasing
        stemmed_tokens = [stemmer.stem(token) for token in tokens]  # Stemming
        vocab_set.update(stemmed_tokens)
    vocab = ['<sos>', '<eos>', '<unk>', '<pad>'] + sorted(list(vocab_set))
    word_to_index = {word: index for index, word in enumerate(vocab)}

    input_data_indices = [[word_to_index.get(stemmer.stem(token), word_to_index['<unk>']) for token in word_tokenize(seq.lower())] for seq in input_seqs]
    target_data_indices = [[word_to_index.get(stemmer.stem(token), word_to_index['<unk>']) for token in word_tokenize(seq.lower())] for seq in target_seqs]

    input_tensor_sequences = [torch.tensor(seq) for seq in input_data_indices]
    target_tensor_sequences = [torch.tensor(seq) for seq in target_data_indices]

    padded_input_sequences = pad_sequence(input_tensor_sequences, batch_first=True, padding_value=word_to_index['<pad>'])
    padded_target_sequences = pad_sequence(target_tensor_sequences, batch_first=True, padding_value=word_to_index['<pad>'])

    train_input, test_input, train_target, test_target = train_test_split(
        padded_input_sequences, padded_target_sequences, train_size=config['split'][0], random_state=42
    )
    val_split = config['split'][1] / (config['split'][1] + config['split'][2])
    train_input, val_input, train_target, val_target = train_test_split(
        train_input, train_target, test_size=val_split, random_state=42
    )

    train_dataset = torch.utils.data.TensorDataset(train_input, train_target)
    val_dataset = torch.utils.data.TensorDataset(val_input, val_target)
    test_dataset = torch.utils.data.TensorDataset(test_input, test_target)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, word_to_index
