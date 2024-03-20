from lstm import LSTM
import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_size, True)  
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out)
        return out
    
    def train_model(model, train_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.transpose(1, 2), targets) 
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate_model(model, loader, criterion, device):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output.transpose(1, 2), targets)
                total_loss += loss.item()
        return total_loss / len(loader)
