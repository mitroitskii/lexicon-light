'''Classes and functions helpful for general model training.
So far used in train_probe.py 
'''
# partially adapted and modified from https://github.com/sfeucht/lexicon as part of collaboration for David Bau's group
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


def _topktoks(hs, k=1):
    _, top_tokens = hs.topk(k=k, dim=-1)
    return top_tokens


def _topkprobs(hs, tokenizer, k=5):
    top_probs, top_tokens = torch.softmax(hs, dim=0).topk(k=k, dim=-1)
    out = {}
    for i in range(k):
        out[f"top_{i+1}_prob"] = top_probs[i].item()
        out[f"top_{i+1}_tok_id"] = top_tokens[i].item()
        out[f"top_{i+1}_tok"] = tokenizer.decode(top_tokens[i].tolist())
    return out


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        output = self.fc(x)
        return output


class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, dropout_rate=0.5):
        super(MLPModel, self).__init__()
        if hidden_size is None:  # default to input size
            hidden_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None):
        super(RNNModel, self).__init__()
        if hidden_size is None:  # default to input size
            hidden_size = input_size
        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Xavier initialization for LSTM weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Xavier initialization for the fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        output, _ = self.rnn(x)
        # output: (batch_size, sequence_length, hidden_size)
        output = self.fc(output[:, -1, :])  # Use the last time step's output
        return output
