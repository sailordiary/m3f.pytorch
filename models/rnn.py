import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .convlstm import BiConvLSTM


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_fcs=1, dropout=False):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

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
        out, _ = self.gru(x)
        if self.num_classes > 0:
            out = self.fc(out)  # predictions based on every time step

        return out


class CLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, fwd_att=False):
        super(CLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.convlstm = BiConvLSTM(input_size=(3, 3), input_dim=input_size, hidden_dim=[hidden_size] * num_layers, kernel_size=(3, 3), num_layers=num_layers, bias=True, return_all_layers=False, fwd_att=fwd_att)
        self.fc = nn.Linear(hidden_size * 3 * 3, num_classes) # nn.Linear(hidden_size*2, num_classes)


    def forward(self, x):
        out = self.convlstm(x)
        out = out.view(out.size(0), out.size(1), -1)
        out = self.fc(out)  # predictions based on every time step

        return out
