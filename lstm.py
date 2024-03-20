from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from collections import Counter
from configs import LSTM_CONFIG
import torch
import torch.nn as nn
import torch.nn.init as init
import math

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
    vocab.append('<pd>')
    vocab.append('<unk>')

    return train, test, val, vocab

train, test, val, vocab = build_dataset('./data/dialogs.txt', LSTM_CONFIG)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, peephole=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.peephole = peephole
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        if peephole:
            self.V = nn.Parameter(torch.Tensor(hidden_size * 3))
        self.init_weights()
    
    def init_weights(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.U)
        init.constant_(self.bias, 0)
        if self.peephole:
            init.constant_(self.V, 0)

    def forward(self, x):
        batch, sequence_size, _ = x.size()
        hidden_sequence = []
        h_t, c_t = (torch.zeros(batch, self.hidden_size).to(x.device), torch.zeros(batch, self.hidden_size).to(x.device))
        HS = self.hidden_size
        for t in range(sequence_size):
            x_t = x[:, t, :]
            
            if self.peephole:
                gates = x_t @ self.U + c_t * self.V + self.bias
            else:
                gates = x_t @ self.U + self.bias
            
            i_t, f_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS*2]),  # forget
                torch.sigmoid(gates[:, HS*3:]),  # output
            )
            
            if self.peephole:
                c_t = f_t * c_t + i_t * torch.tanh(gates[:, HS*2:HS*3])
            else:
                c_t = f_t * c_t + i_t * torch.tanh(gates[:, HS*2:HS*3])
                
            h_t = o_t * torch.tanh(c_t)
                
            hidden_sequence.append(h_t.unsqueeze(0))
            
        hidden_sequence = torch.cat(hidden_sequence, dim=0)
        hidden_sequence = hidden_sequence.transpose(0, 1).contiguous() # reshape (sequence, batch, feature) to (batch, sequence, feature)
        
        return hidden_sequence, (h_t, c_t)
