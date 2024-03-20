import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, peephole=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.peephole = peephole
        
        self.W = nn.Linear(input_size, hidden_size * 4)  # For input gate, forget gate, cell gate, and output gate
        self.U = nn.Linear(hidden_size, hidden_size * 4, bias=False)  # For hidden state

        if peephole:
            self.V_i = nn.Parameter(torch.Tensor(hidden_size))  # Peephole for input gate
            self.V_f = nn.Parameter(torch.Tensor(hidden_size))  # Peephole for forget gate
            self.V_o = nn.Parameter(torch.Tensor(hidden_size))  # Peephole for output gate
            nn.init.constant_(self.V_i, 0)
            nn.init.constant_(self.V_f, 0)
            nn.init.constant_(self.V_o, 0)
    
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hidden_sequence = []
        
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        for t in range(sequence_size):
            x_t = x[:, t, :]
            
            gates = self.W(x_t) + self.U(h_t)
            i_t, f_t, g_t, o_t = gates.chunk(4, 1)
            
            if self.peephole:
                i_t = torch.sigmoid(i_t + self.V_i * c_t)
                f_t = torch.sigmoid(f_t + self.V_f * c_t)
            else:
                i_t = torch.sigmoid(i_t)
                f_t = torch.sigmoid(f_t)
            
            g_t = torch.tanh(g_t)
            
            c_t_new = f_t * c_t + i_t * g_t
            
            if self.peephole:
                o_t = torch.sigmoid(o_t + self.V_o * c_t_new)
            else:
                o_t = torch.sigmoid(o_t)
            
            h_t = o_t * torch.tanh(c_t_new)
            c_t = c_t_new
            
            hidden_sequence.append(h_t.unsqueeze(0))
        
        hidden_sequence = torch.cat(hidden_sequence, dim=0)
        hidden_sequence = hidden_sequence.transpose(0, 1)
        
        return hidden_sequence, (h_t, c_t)
