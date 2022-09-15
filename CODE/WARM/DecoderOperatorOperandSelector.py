import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class DecoderOperatorOperandSelector(nn.Module):
    def __init__(self, hidden_size, output_size_operator, output_size_operand, device, n_layers=1, dropout=0.0):
        super(DecoderOperatorOperandSelector, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size_operator, 64)
        self.em_dropout = nn.Dropout(self.dropout)
        self.gru_operator = nn.GRU(64, hidden_size, num_layers=self.n_layers, dropout=self.dropout)
        self.out_operator = nn.Linear(hidden_size, output_size_operator)
        self.gru_operand_1 = nn.GRU(64, hidden_size, num_layers=self.n_layers, dropout=self.dropout)
        self.out_operand_1 = nn.Linear(hidden_size, output_size_operand)
        self.gru_operand_2 = nn.GRU(64, hidden_size, num_layers=self.n_layers, dropout=self.dropout)
        self.out_operand_2 = nn.Linear(hidden_size, output_size_operand)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    '''
    input: b * t
    hidden: 1 * b * hidden_dim
    '''
    def forward(self, input, hidden, sample=False):
        output = self.embedding(input.T)#.view(1, 1, -1)
        output = self.em_dropout(output)
        output = F.relu(output)
        output_operator, hidden = self.gru_operator(output, hidden)
        output_operator = self.softmax(self.out_operator(output_operator[0]))
        if sample:
            operator = torch.multinomial(output_operator, 1).T
        else:
            operator = torch.argmax(output_operator, 1).unsqueeze(0)
        operator_embedding = self.embedding(operator)#.view(1, 1, -1)
        operator_embedding = self.em_dropout(operator_embedding)
        operator_embedding = F.relu(operator_embedding)
        output_operand_1, hidden = self.gru_operand_1(operator_embedding, hidden)
        output_operand_1 = self.softmax(self.out_operand_1(output_operand_1[0]))
        output_operand_2, hidden = self.gru_operand_2(operator_embedding, hidden)
        output_operand_2 = self.softmax(self.out_operand_2(output_operand_2[0]))
        return output_operator, output_operand_1, output_operand_2, hidden, operator

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)