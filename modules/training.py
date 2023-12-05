'''Classes and functions helpful for general model training.
So far used in train_probe.py 
'''
# partially adapted and modified from https://github.com/sfeucht/lexicon as part of collaboration for David Bau's group
import torch
import torch.nn as nn
import numpy as np


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
        # x: (batch_size, input_size)
        output = self.shift_embeddings(x)
        output, _ = self.rnn(output)
        # output: (sequence_length, batch_size, hidden_size)
        output = self.fc(output[-1, :, :])  # Use the last time step's output
        return output

    def shift_embeddings(self, embeddings):
        # 1. Add Noise
        noise = torch.randn_like(embeddings) * 0.05  # Small noise
        noised_embeddings = embeddings + noise

        # 2. Apply Rotation
        rotated_embeddings = self.rotate_embeddings(embeddings)

        # 3. Interpolate Embeddings
        if embeddings.size(0) == 1:  # if batch size is 1
            interpolated_embeddings = (
                embeddings + noised_embeddings + rotated_embeddings) / 3
        else:
            interpolated_embeddings = self.interpolate_embeddings(embeddings)

        # Concatenate altered embeddings over a new sequence dimension
        return torch.stack([noised_embeddings, rotated_embeddings, interpolated_embeddings], dim=0)

    def interpolate_embeddings(self, embeddings):
        # Random Embedding Selection for Interpolation
        batch_size = embeddings.size(0)
        indices = torch.randperm(batch_size).to(
            embeddings.device)  # Random permutation of indices
        # Select embeddings based on random indices
        random_embeddings = embeddings[indices]

        # Ensure not to interpolate an embedding with itself
        for i in range(batch_size):
            if indices[i] == i:
                # If selected embedding is the same, swap with the next one (or previous one for the last element)
                swap_with = i + 1 if i < batch_size - 1 else i - 1
                random_embeddings[i], random_embeddings[swap_with] = random_embeddings[swap_with], random_embeddings[i]

        # Interpolation
        return (embeddings + random_embeddings) / 2

    def rotate_embeddings(self, embeddings):
        # Identify the device of the embeddings
        device = embeddings.device
        # Generate random matrix
        W = torch.randn(embeddings.size(-1), embeddings.size(-1), device=device)
        # Decompose to get orthogonal matrix
        Q, _ = torch.linalg.qr(W)
        # Rotate embeddings
        return embeddings @ Q


