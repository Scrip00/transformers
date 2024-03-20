import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from lstmmodel import LSTMModel 
from utils import build_dataset
from configs import LSTM_CONFIG

def main(model):
    train_loader, test_loader, val_loader, vocab = build_dataset(LSTM_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = LSTM_CONFIG['embed_dim']
    hidden_size = LSTM_CONFIG['hidden_dim']
    model = LSTMModel(len(vocab), embedding_dim, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LSTM_CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    num_epochs = LSTM_CONFIG['epochs']
    for epoch in range(num_epochs):
        train_loss = model.train_model(train_loader, optimizer, criterion, device)
        val_loss = model.evaluate_model(val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    test_loss = model.evaluate_model(test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')
    torch.save(model, 'lstm.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to demonstrate command line argument parsing.")
    parser.add_argument("--model", type=str, required=True, help="Specify the model to use.")
    args = parser.parse_args()
    main(args.model)
