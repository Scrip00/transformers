import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from lstmmodel import LSTMModel
from utils import build_dataset 
from configs import LSTM_CONFIG

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def prepare_sequence(seq, vocab, device):
    idxs = [vocab[word] if word in vocab else vocab["<unk>"] for word in seq.split()]
    return torch.tensor(idxs, dtype=torch.long).to(device)

def top_k_sampling(logits, k=5):
    logits_shape = logits.shape
    logits = logits.view(-1, logits_shape[-1])  # Reshape to [batch_size * num_classes]
    sorted_indices = torch.argsort(logits, descending=True)
    top_k_indices = sorted_indices[:, :k]
    top_k_logits = logits.gather(dim=1, index=top_k_indices)
    probabilities = F.softmax(top_k_logits, dim=-1)
    probabilities = probabilities.view(logits_shape[0], -1, k)  # Reshape back to [batch_size, num_classes, k]
    sampled_index = torch.multinomial(probabilities.squeeze(), num_samples=1)
    return top_k_indices.gather(dim=1, index=sampled_index.unsqueeze(dim=-1)).squeeze()


def predict_sequence(model, vocab, idx_to_word, input_seq, device, eos_token='<eos>', max_length=50, temperature=0.7, k=20):
    generated_sequence = input_seq.split() if input_seq else [] 
    current_input = input_seq

    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            inputs = prepare_sequence(current_input, vocab, device)
            inputs = pad_sequence([inputs], batch_first=True, padding_value=vocab['<pad>'])
            outputs = model(inputs)
            logits = outputs[:, -1, :] / temperature 
            sampled_index = top_k_sampling(logits, k=k)
            predicted_index = sampled_index.item()

            predicted_word = idx_to_word[predicted_index]

            if predicted_word == eos_token or predicted_word == '<pad>':
                print("LOL")
                break

            generated_sequence.append(predicted_word)
            current_input = ' '.join(generated_sequence)

    return ' '.join(generated_sequence)

def chat(model, vocab, idx_to_word, device):
    print("Chatbot activated. Type 'quit' to exit.")
    while True:
        input_seq = input("You: ")
        if input_seq.lower() == 'quit':
            break
        response = predict_sequence(model, vocab, idx_to_word, input_seq, device)
        print(f"Bot: {response[len(input_seq):]}")

def main():
    model_path = 'lstm.pth'

    _, _, _, vocab = build_dataset(LSTM_CONFIG)
    idx_to_word = {idx: word for word, idx in vocab.items()}
    model = load_model(model_path)
    device = next(model.parameters()).device  

    chat(model, vocab, idx_to_word, device)

if __name__ == "__main__":
    main()
