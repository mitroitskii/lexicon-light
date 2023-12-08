'''Classes and functions for caching hidden states used as training input. 
'''
# partially adapted from https://github.com/sfeucht/lexicon and modified as part of a research project at David Bau's lab

import torch
import baukit
import numpy as np
from torch.utils.data import Dataset


class DocDataset(Dataset):
    '''
    Dataset that retrieves hidden representations for a document as a "slice" of a model,
    i.e. for a given layer, it retrieves all the hidden states for document tokens at that layer, as well as all the tokens for that document. 

    Labels are one-hot encodings of the CURRENT token (i.e. tokens[i] is the one-hot encoding
    of the token that hidden_states[i] encodes). For train_probe.py, target_idx handling
    happens in the collate function when we're loading the data.
    '''

    def __init__(self, model, tokenizer, model_name, layer, target_idx, dataset_csv, window_size, vocab_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        # int: -1 is embedding, 0-27 for layers, 28 for logits right at the end
        self.layer = layer
        self.target_idx = target_idx  # -1 is previous token, 0 is current.
        self.dataset_csv = dataset_csv
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.device = device

        if self.target_idx not in [0, -1]:
            raise Exception(
                f"target_idx {self.target_idx} not yet supported (doc_collate will not work).")

    # when tokenizing make sure to include BOS token at beginning as "padding"
    def tokenize(self, text):
        bos = self.tokenizer.bos_token
        t = self.tokenizer(bos+text)

        if len(t) > self.window_size:
            return t[:self.window_size]  # truncate to window size
        else:
            return t

    def get_model_trace(self, text):
        # Convert the input text into tensors and move them to the device (GPU/CPU) where the model is located
        inp = {k: torch.tensor(v)[None].to(self.device)
               for k, v in self.tokenize(text).items()}

        # Get the names of the layers in the model that match the given regular expression
        layers = [n for n, _ in self.model.named_modules() if re.match(
            r'^transformer.wte|^transformer.h.\d+$|^ln_f', n)]

        # Use baukit.Trace to trace the execution of the model on the specified layer
        with baukit.Trace(self.model, layers[self.layer+1]) as tr:
            # Run the model with the input tensors
            _ = self.model(**inp)

        # Get the input ids from the input tensors
        inp = inp['input_ids']

        # Return the input ids, the output from the traced layer
        return inp, tr.output[0]

    def __len__(self):
        return len(self.dataset_csv)  # number of docs, not tokens

    def __getitem__(self, index):
        # Access the row at the given index in the dataset CSV file
        doc = self.dataset_csv.iloc[index]

        # 'decoded_prefix' is a column in the CSV that contains the single input from the pile dataset gpt-j was trained on.
        # The method returns two values: input ids and the hidden states from the traced layer
        inp, trout = self.get_model_trace(str(doc['decoded_prefix']))

        # Squeeze the trout tensor to remove dimensions of size one from the shape of the tensor.
        # The resulting tensor, hidden_states, has dimensions (doc_length, model_dim)
        hidden_states = trout.squeeze()

        # Convert the token indices (inp) into one-hot encodings.
        # np.eye creates an identity matrix of size self.vocab_size.
        # Indexing this matrix with inp.cpu() selects the rows corresponding to the token indices, creating one-hot vectors.
        # These vectors are then converted to a PyTorch tensor and moved to the CPU.
        # The resulting tensor is squeezed to remove dimensions of size one.
        targets = torch.tensor(np.eye(self.vocab_size)[
                               inp.cpu()], device='cpu').squeeze()

        # Assert that the lengths of hidden_states and targets are equal.
        # This is a sanity check to ensure that the number of hidden states matches the number of target tokens.
        assert len(hidden_states) == len(targets)

        # Return a tuple containing three elements:
        # 1) a tensor containing the index,
        # 2) the hidden_states tensor,
        # 3) the targets one hot tensor
        return torch.tensor(index), hidden_states, targets


class DocCollate(object):
    '''
    This collate function is meant to be used with the DocDataset. Has to be a class so that target_idx can be specified. 

    Collate function for predicting the previous token from a current hidden state.
    Flattens a batch of several documents' hidden states with their corresponding tokens
    into single arrays of current hidden_states, previous_tokens, current_tokens, and doc_idxs
    (so that document information can be recovered during testing). 

    "total_batch_tokens" refers to the sum of the lengths of each document in the given batch (whereas the length of each document is < WINDOW_SIZE). 
    Each output array is of length total_batch_tokens.

    Parameters:
        batch: list of tuples, where each tuple represents a document. for example, 
            (0, hidden_states, tokens) contains the hidden states and corresponding
            tokens for document 0 in the dataset. document index 0 indicates the first
            document in the entire dataset, not just in that batch.

    Returns:
        current_hss: current hidden states, shape (total_batch_tokens, model_dim)
        target_toks: one-hot embeddings for previous tokens, shape (total_batch_tokens, vocab_size)
        current_toks: one-hot embeddings for current tokens, shape (total_batch_tokens, vocab_size)
        doc_idxs: document indices each hidden state came from, shape (total_batch_tokens,)
    '''

    def __init__(self, target_idx):
        self.target_idx = target_idx

        if self.target_idx not in [0, -1]:
            raise Exception(
                f"target_idx {self.target_idx} not yet supported (DocCollate will not work).")

    def __call__(self, batch):
        current_hss = []
        target_toks = []
        current_toks = []
        doc_idxs = []
        for doc in batch:
            # batch looks like [doc0:(0, hidden_states, tokens), doc1:(1, hidden_states, tokens)...]
            doc_idx, hidden_states, tokens = (a.cpu() for a in doc)

            # take all of the hidden states after BOS token, and all of the token labels except for the last one
            # current_hss: BOS [this is an example sentence]
            # target_toks: [BOS this is an example] sentence
            if self.target_idx == -1:
                current_hss.append(hidden_states[1:])
                target_toks.append(tokens[:-1])
                current_toks.append(tokens[1:])
                doc_idxs.append(torch.tensor(
                    [doc_idx for _ in range(len(hidden_states[1:]))], device='cpu'))

            # exclude predicting bos_embedding -> BOS
            elif self.target_idx == 0:
                current_hss.append(hidden_states[1:])
                target_toks.append(tokens[1:])
                current_toks.append(tokens[1:])
                doc_idxs.append(torch.tensor(
                    [doc_idx for _ in range(len(hidden_states[1:]))], device='cpu'))

        current_hss = torch.cat(current_hss)
        target_toks = torch.cat(target_toks)
        current_toks = torch.cat(current_toks)
        doc_idxs = torch.cat(doc_idxs)

        return (current_hss, target_toks, current_toks, doc_idxs)


class AllEmbeds(torch.utils.data.Dataset):
    def __init__(self, model, tokenizer, model_name, window_size, vocab_size, device):
        """
        Initialize the AllEmbeds dataset.

        The length of this dataset is equal to the vocabulary size.
        Indexing this dataset returns the index itself.

        Args:
            model (torch.nn.Module): The model used for embedding.
            vocab_size (int): The size of the vocabulary.
            device (torch.device): The device to run the model on.
        """
        self.model = model
        self.vocab_size = vocab_size
        self.device = device

        self.layer_name = -1
        self.target_idx = 0



    def __len__(self):
        return self.vocab_size

    def __getitem__(self, index):
        return index

    def allembeds_collate(self, batch):
        """
        Collates a batch of indexes and returns the corresponding embeddings and one-hot encodings.

        Args:
            batch (list): A list of indexes.

        Returns:
            tuple: A tuple containing the embeddings and one-hot encodings.
        """
        inp = torch.tensor(batch).to(self.device)
        embed = self.model.transformer.wte(inp).squeeze().cpu().float()

        onehot = torch.tensor(np.eye(self.vocab_size)[batch], device='cpu')
        return embed, onehot
