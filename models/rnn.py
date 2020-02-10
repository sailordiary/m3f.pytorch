import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_fcs=1, dropout=False, return_h=False):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.return_h = return_h

        if self.num_classes > 0:
            if num_fcs == 1:
                self.fc = nn.Linear(hidden_size*2, num_classes)
            elif num_fcs == 2:
                if dropout:
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size*2, hidden_size),
                        nn.ReLU(True),
                        nn.Dropout(0.5),
                        nn.Linear(hidden_size, num_classes)
                    )
                else:
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size*2, hidden_size),
                        nn.ReLU(True),
                        nn.Linear(hidden_size, num_classes)
                    )
            elif num_fcs == 3:
                if dropout:
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size*2, hidden_size),
                        nn.ReLU(True),
                        nn.Dropout(0.5),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(True),
                        nn.Dropout(0.5),
                        nn.Linear(hidden_size, num_classes)
                    )
                else:
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size*2, hidden_size),
                        nn.ReLU(True),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(True),
                        nn.Linear(hidden_size, num_classes)
                    )

        # init
        stdv = math.sqrt(2 / (input_size + hidden_size))
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                for i in range(0, hidden_size * 3, hidden_size):
                    nn.init.uniform_(param.data[i: i + hidden_size],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
            elif 'weight_hh' in name:
                for i in range(0, hidden_size * 3, hidden_size):
                    nn.init.orthogonal_(param.data[i: i + hidden_size])
            elif 'bias' in name:
                for i in range(0, hidden_size * 3, hidden_size):
                    nn.init.constant_(param.data[i: i + hidden_size], 0)

    def forward(self, x):
        self.gru.flatten_parameters()
        # x_lens = torch.LongTensor(lens)
        # x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        out, h = self.gru(x)
        if self.num_classes > 0:
            out = self.fc(out)  # predictions based on every time step
        if self.return_h:
            return out, h
        else:
            return out


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1) # B*T*H
        # encoder_outputs: [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size=128, hidden_size=512, output_size=2,
                 n_layers=1):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs, last_hidden, encoder_outputs):
        # inputs: B*2, last_hidden: B*T*H, enc_outputs: B*T*H
        inputs = inputs.unsqueeze(1)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        # B*1*T bmm B*T*H -> B*1*H
        context = attn_weights.bmm(encoder_outputs)  # (B,1,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([inputs, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)  # (B,1,N) -> (B,N)
        context = context.squeeze(1)
        output = self.out(torch.cat([output, context], 1)) # (B, 2)
        return output, hidden, attn_weights


class AttEncDec(nn.Module):
    def __init__(self):
        super(AttEncDec, self).__init__()
        self.encoder = GRU(1024, 512, 2, -1, return_h=True)
        self.decoder = Decoder(2, 512, 2, 1)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        if trg is not None:
            max_len = trg.size(1)
        else:
            max_len = src.size(1)
        outputs = src.new_zeros(batch_size, max_len, 2)

        encoder_output, hidden = self.encoder(src)
        # sum bidirectional outputs
        encoder_output = encoder_output[:, :, :512] + encoder_output[:, :, 512:]
        hidden = hidden[: self.decoder.n_layers]
        output = src.new_zeros(batch_size, 2)
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[:, t] = output
            is_teacher = trg is not None and random.random() < teacher_forcing_ratio
            output = trg.data[:, t] if is_teacher else output
            output = output.to(src.device)
        return outputs
