import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderFCN2(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size_operator, output_size_operand, device, dropout=0.5):
        super(DecoderFCN2, self).__init__()
        self.dropout = dropout

        self.em_dropout = nn.Dropout(self.dropout)

        # self.attn = Attn(hidden_size)

        self.embedding_operator = nn.Embedding(output_size_operator, embedding_size)
        self.gen_operator = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.gen_operator_g = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.gen_operator_2 = nn.Linear(hidden_size, hidden_size)
        self.gen_operator_2_g = nn.Linear(hidden_size, hidden_size)
        # self.gen_operator_concat = nn.Linear(hidden_size*2, hidden_size)
        self.gen_output_operator = nn.Linear(hidden_size, output_size_operator)

        self.embedding_operand = nn.Embedding(output_size_operand, embedding_size)
        self.gen_operand_1 = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.gen_operand_1_g = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.gen_operand_1_2 = nn.Linear(hidden_size, hidden_size)
        self.gen_operand_1_2_g = nn.Linear(hidden_size, hidden_size)
        # self.gen_operand_1_concat = nn.Linear(hidden_size*2, hidden_size)
        self.gen_output_operand_1 = nn.Linear(hidden_size, output_size_operand)

        self.gen_operand_2 = nn.Linear(hidden_size*2 + embedding_size, hidden_size)
        self.gen_operand_2_g = nn.Linear(hidden_size*2 + embedding_size, hidden_size)
        self.gen_operand_2_2 = nn.Linear(hidden_size, hidden_size)
        self.gen_operand_2_2_g = nn.Linear(hidden_size, hidden_size)
        # self.gen_operand_2_concat = nn.Linear(hidden_size*2, hidden_size)
        self.gen_output_operand_2 = nn.Linear(hidden_size, output_size_operand)

    def forward(self, input, hidden, sample=False, seq_mask=None):
        output = self.embedding_operator(input.T)  # .view(1, 1, -1)
        output = self.em_dropout(output).squeeze()

        if hidden.squeeze(0).dim() != output.dim() and hidden.ndim > 2: hidden = hidden.sum(0)
        if hidden.dim() != output.dim(): hidden = hidden.squeeze(0)
        out_cat = torch.cat((output, hidden), 1)

        # attn_weights = self.attn(hidden.unsqueeze(0), inputs, seq_mask)
        # context = attn_weights.bmm(inputs.transpose(0, 1))  # B x S=1 x N

        # out_cat = torch.cat((output, hidden.squeeze()), 1)
        # out_cat = torch.cat((output, context.squeeze()), 1)

        output_operator_hidden = torch.tanh(self.gen_operator(out_cat))
        output_operator_hidden_g = torch.sigmoid(self.gen_operator_g(out_cat))
        output_operator_hidden = output_operator_hidden * output_operator_hidden_g
        output_operator_hidden_2 = torch.tanh(self.gen_operator_2(output_operator_hidden))
        output_operator_hidden_2_g = torch.sigmoid(self.gen_operator_2_g(output_operator_hidden))
        output_operator_hidden_2 = output_operator_hidden_2 * output_operator_hidden_2_g
        # output_operator_hidden_2 = torch.tanh(self.gen_operator_concat(torch.cat((output_operator_hidden_2, context.squeeze()), 1)))
        output_operator = self.gen_output_operator(output_operator_hidden_2)

        output_operator = F.softmax(output_operator, dim=1)
        if sample:
            operator = torch.multinomial(output_operator, 1).T
        else:
            operator = torch.argmax(output_operator, 1).unsqueeze(0)

        operator_embedding = self.embedding_operator(operator)
        operator_embedding = self.em_dropout(operator_embedding)


        out_cat = torch.cat((operator_embedding.squeeze(), output_operator_hidden), 1)
        output_operand_1_hidden = torch.tanh(self.gen_operand_1(out_cat))
        output_operand_1_hidden_g = torch.sigmoid(self.gen_operand_1_g(out_cat))
        output_operand_1_hidden = output_operand_1_hidden * output_operand_1_hidden_g
        output_operand_1_hidden_2 = torch.tanh(self.gen_operand_1_2(output_operand_1_hidden))
        output_operand_1_hidden_2_g = torch.sigmoid(self.gen_operand_1_2_g(output_operand_1_hidden))
        output_operand_1_hidden_2 = output_operand_1_hidden_2 * output_operand_1_hidden_2_g
        # output_operand_1_hidden_2 = torch.tanh(self.gen_operand_1_concat(torch.cat((output_operand_1_hidden_2, context.squeeze()), 1)))
        output_operand_1 = self.gen_output_operand_1(output_operand_1_hidden_2)

        out_cat = torch.cat((operator_embedding.squeeze(), output_operand_1_hidden, output_operator_hidden), 1)
        output_operand_2_hidden = torch.tanh(self.gen_operand_2(out_cat))
        output_operand_2_hidden_g = torch.sigmoid(self.gen_operand_2_g(out_cat))
        output_operand_2_hidden = output_operand_2_hidden * output_operand_2_hidden_g
        output_operand_2_hidden_2 = torch.tanh(self.gen_operand_2_2(output_operand_2_hidden))
        output_operand_2_hidden_2_g = torch.sigmoid(self.gen_operand_2_2_g(output_operand_2_hidden))
        output_operand_2_hidden_2 = output_operand_2_hidden_2 * output_operand_2_hidden_2_g
        # output_operand_2_hidden_2 = torch.tanh(self.gen_operand_2_concat(torch.cat((output_operand_2_hidden_2, context.squeeze()), 1)))
        output_operand_2 = self.gen_output_operand_1(output_operand_2_hidden_2)

        return output_operator, output_operand_1, output_operand_2, output_operand_2_hidden, operator
