import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GruEncoder(nn.Module):
    def __init__(self, encoder_hidden_size, embed_size, feat_vec_size, vocab_size, pretrained_emb=None,
                 bidirectional=True, n_layers=1, dropout=0.0):
        super(GruEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = encoder_hidden_size
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.feat_vec_size = feat_vec_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.feat_lookup = nn.Embedding(2, self.feat_vec_size, padding_idx=0)
        self.lookup = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)
        self.lookup_dropout = nn.Dropout(dropout)

        if pretrained_emb is not None:
            self.lookup.weight = nn.Parameter(pretrained_emb)
            self.lookup.weight.requires_grad = False

        self.encoder_gru = nn.GRU(self.embed_size + self.feat_vec_size, self.hidden_size, bidirectional=self.bidirectional,
                               num_layers=n_layers, dropout=dropout)

    def forward(self,input,lengths,features,sorted=False):
        """
        :param input: tensor (max_len * batch_size)
        :param lengths: tensor (batch_size)
        :param features: tensor (max_len * batch_size)
        :param sorted: bool
        :return output: tensor (max_len * batch_size * (2xhidden_size))
        :return state: tensor (2 * max_len * batch_size)
        """
        embedded = self.lookup(input)
        embedded = self.lookup_dropout(embedded)
        sorted_lengths = lengths

        feat_embed = self.feat_lookup(features)
        embedded = torch.cat((embedded, feat_embed), 2)

        if not sorted:
            sorted_lengths, sorted_indices = torch.sort(lengths, dim=0, descending=True)
            embedded = embedded[:, sorted_indices, :]
            orig_indices = torch.zeros(lengths.size(0),dtype=torch.long)
            for i, val in enumerate(sorted_indices):
                orig_indices[val] = i

        embedded = pack_padded_sequence(embedded, sorted_lengths)
        output, state = self.encoder_gru(embedded)
        output, _ = pad_packed_sequence(output)

        if not sorted:
            output = output[:, orig_indices, :]
            state = state[:, orig_indices, :]

        encoder_out = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        encoder_state = state.view(self.n_layers, self.directions, -1, self.hidden_size).sum(1)
        
        return encoder_out, encoder_state