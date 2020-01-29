import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# from allennlp.modules.input_variational_dropout import InputVariationalDropout
# from allennlp.nn.initializers import block_orthogonal

from .convlstm import BiConvLSTM


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        x, lengths = pad_packed_sequence(x, batch_first=True)
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class RNNDropout(nn.Module):
    def __init__(self, p):
        self.layer_dropout = InputVariationalDropout(p)
    
    def forward(self, pack):
        seq, lengths = pad_packed_sequence(pack, batch_first=True)
        output_sequence = self.layer_dropout(seq)
        output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=True)
        
        return output_sequence


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, use_dropout=False, use_bn=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        
        lstm_fw = []
        lstm_bw = []
        for i in range(num_layers):
            if self.use_dropout:
                lstm_fw.append(RNNDropout(0.30))
                lstm_bw.append(RNNDropout(0.30))
            if self.use_bn:
                lstm_fw.append(SequenceWise(nn.BatchNorm1d(input_size if i==0 else hidden_size)))
                lstm_bw.append(SequenceWise(nn.BatchNorm1d(input_size if i==0 else hidden_size)))
            lstm_fw.append(nn.LSTM(input_size if i == 0 else hidden_size, hidden_size, batch_first=True))
            lstm_bw.append(nn.LSTM(input_size if i == 0 else hidden_size, hidden_size, batch_first=True))
        self.lstm_fw = nn.Sequential(*lstm_fw)
        self.lstm_bw = nn.Sequential(*lstm_bw)
        
        self.last_dropout = nn.Dropout(0.15)
        self.bn = nn.BatchNorm1d(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.lstm_fw.modules():
            if isinstance(m, nn.LSTM):
                block_orthogonal(m.weight_ih_l0.data, [m.hidden_size, m.input_size])
                block_orthogonal(m.weight_hh_l0.data, [m.hidden_size, m.hidden_size])

                m.bias_ih_l0.data.fill_(0.)
                m.bias_hh_l0.data.fill_(0.)
                # Initialize forget gate biases to 1.0 as per An Empirical
                # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
                m.bias_ih_l0.data[m.hidden_size:2 * m.hidden_size].fill_(1.0)
        for m in self.lstm_bw.modules():
            if isinstance(m, nn.LSTM):
                block_orthogonal(m.weight_ih_l0.data, [m.hidden_size, m.input_size])
                block_orthogonal(m.weight_hh_l0.data, [m.hidden_size, m.hidden_size])

                m.bias_ih_l0.data.fill_(0.)
                m.bias_hh_l0.data.fill_(0.)
                m.bias_hh_l0.data[m.hidden_size:2 * m.hidden_size].fill_(1.0)

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                m.flatten_parameters()
        lengths = torch.IntTensor([29] * x.size(0))
        
        x_rev = torch.flip(x, [1]) # TODO: can't do it like this with variable length inputs
        x = pack_padded_sequence(x, lengths, batch_first=True)
        fw_out, _ = self.lstm_fw(x)
        fw_out, _ = pad_packed_sequence(fw_out, batch_first=True)
        
        x_rev = pack_padded_sequence(x_rev, lengths, batch_first=True)
        bw_out_rev, _ = self.lstm_bw(x_rev)
        bw_out_rev, _ = pad_packed_sequence(bw_out_rev, batch_first=True)
        bw_out = torch.flip(bw_out_rev, [1])
        out = torch.cat((fw_out, bw_out), 2)
        
        if self.use_dropout:
            out = self.last_dropout(out)
        out = torch.mean(out, 1) # temporal avgpool
        if self.use_bn:
            out = self.bn(out)
        out = self.fc(out)
        
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

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
        out = self.fc(out)  # predictions based on every time step

        return out


class CLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, fwd_att=False):
        super(CLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame
        self.convlstm = BiConvLSTM(input_size=(3, 3), input_dim=input_size, hidden_dim=[hidden_size] * num_layers, kernel_size=(3, 3), num_layers=num_layers, bias=True, return_all_layers=False, fwd_att=fwd_att)
        self.fc = nn.Linear(hidden_size * 3 * 3, num_classes) # nn.Linear(hidden_size*2, num_classes)


    def forward(self, x):
        out = self.convlstm(x)
        out = out.view(out.size(0), out.size(1), -1)
        out = self.fc(out)  # predictions based on every time step

        return out

