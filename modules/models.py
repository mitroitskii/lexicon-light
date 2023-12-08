'''Models used for training probes'''
# partially adapted and modified from https://github.com/sfeucht/lexicon as part of collaboration for David Bau's group
import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """
    Linear model class that represents a linear transformation of the input data.

    Args:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.

    Attributes:
        fc (nn.Linear): The linear layer for the transformation.

    Methods:
        forward(x): Performs the forward pass of the linear model.

    """

    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """
        Performs the forward pass of the linear model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        output = self.fc(x)
        return output


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron Model.

    Args:
        input_size (int): The size of the input features.
        output_size (int): The size of the output.
        hidden_size (int, optional): The size of the hidden layer. Defaults to None.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.5.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        relu (nn.ReLU): The ReLU activation function.
        dropout (nn.Dropout): The dropout layer.
        fc2 (nn.Linear): The second fully connected layer.

    Methods:
        forward(x): Performs the forward pass of the MLP model.
    """

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
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output


class RNNModel(nn.Module):
    """
    RNNModel is a PyTorch module that implements a recurrent neural network model.

    Args:
        input_size (int): The size of the input features.
        output_size (int): The size of the output.
        hidden_size (int, optional): The size of the hidden state. If not provided, it defaults to the input size.

    Attributes:
        rnn (nn.LSTM): The LSTM layer used for the recurrent computation.
        fc (nn.Linear): The fully connected layer used for the final output.

    Methods:
        forward(x): Performs a forward pass through the model.
        shift_embeddings(embeddings): Applies noise, rotation, and interpolation to the input embeddings.
        interpolate_embeddings(embeddings): Interpolates the input embeddings with randomly selected embeddings.
        rotate_embeddings(embeddings): Rotates the input embeddings using a random orthogonal matrix.
    """

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
        """
        Performs forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            output: Output tensor of shape (batch_size, hidden_size).
        """
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
        W = torch.randn(embeddings.size(-1),
                        embeddings.size(-1), device=device)
        # Decompose to get orthogonal matrix
        Q, _ = torch.linalg.qr(W)
        # Rotate embeddings
        return embeddings @ Q
