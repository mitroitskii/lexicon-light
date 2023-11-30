'''Functionality for caching hidden states of a model representing specific docs,
and using those hidden states for training probes. 
'''
# adapted from https://github.com/sfeucht/lexicon as part of collaboration for David Bau's group

import os
import re
import shutil
import torch
import baukit
import numpy as np
from torch.utils.data import Dataset


'''
Dataset that retrieves a single hidden state, along with its token_id and
the token_id corresponding to target_idx (e.g. the token_id of the token before it
when target_idx=-1). Tokens are ints.
'''
class HiddenStateDataset(Dataset):
    def __init__(self, model, tokenizer, model_name, layer_name, target_idx, dataset_csv, window_size, vocab_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.layer_name = layer_name # int: -1 is embedding, 0-31 for layers, 32 for logits right at the end
        self.target_idx = target_idx # -1 is previous token, 0 is current. 
        self.dataset_csv = dataset_csv
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.device = device

        if self.target_idx not in [0, -1]:
            raise Exception(f"target_idx {self.target_idx} not yet supported.")

        self.doc_lengths = -1 * np.ones(len(self.dataset_csv), dtype=int)
        for idx, doc in baukit.pbar(self.dataset_csv.iterrows()):
            self.doc_lengths[idx] = len(self.tokenize(str(doc['decoded_prefix']))) - 1 # subtract BOS
        assert((self.doc_lengths > 0).all())
    
    # when tokenizing make sure to include BOS token at beginning as "padding"
    def tokenize(self, text):
        if "llama" in self.model_name:
            t = torch.tensor(self.tokenizer.encode(text, bos=True, eos=False))
        else:
            bos = self.tokenizer.bos_token
            t = self.tokenizer(bos+text)
        
        if len(t) > self.window_size:
            return t[:self.window_size]
        else:
            return t
    
    def get_model_trace(self, text, idx):
        if "llama" in self.model_name:
            tokenized = self.tokenize(text)
            self.doc_lengths[idx] = len(tokenized) - 1 # subtract BOS
            inp = tokenized.unsqueeze(0).to(self.device)
            
            layers = [n for n, _ in self.model.named_modules() if re.match(r'^tok_embeddings|^layers.\d+$|^norm', n)]
            with baukit.Trace(self.model, layers[self.layer_name+1]) as tr:
                _ = self.model(inp, 0) # (bsz=1, doc_length, vocab_size)
                
            return inp, tr.output, layers
        
        else:
            self.doc_lengths[idx] = len(self.tokenize(text)['input_ids']) - 1 # subtract BOS
            inp = {k: torch.tensor(v)[None].to(self.device) for k, v in self.tokenize(text).items()}
            layers = [n for n, _ in self.model.named_modules() if re.match(r'^transformer.wte|^transformer.h.\d+$|^transformer.ln_f', n)]
            with baukit.Trace(self.model, layers[self.layer_name+1]) as tr:
                _ = self.model(**inp)
            inp = inp['input_ids']
        
            return inp, tr.output[0], layers
    
    def tokenidx_to_docidx(self, token_idx):
        # if doc0 is len 5, then its token_idxs are <bos> 0 1 2 3 4
        # if doc1 is len 4, then its token_idxs are <bos> 5 6 7 8
        # if doc2 is len 7, then its token_idxs are <bos> 9 10 11 12 13 14 15
        # cum = [5, 9, 16]
        cum = np.cumsum(self.doc_lengths)
        for doc_idx, bound in enumerate(cum):
            if token_idx < bound: 
                relative_idx = token_idx - cum[doc_idx-1] if doc_idx>0 else token_idx
                return doc_idx, relative_idx + 1 # shift to remove BOS
        raise Exception(f"token_idx {token_idx} lies out of bounds")

    def __len__(self):
        return int(sum(self.doc_lengths)) # number of tokens, not docs

    def __getitem__(self, index):
        # index is a TOKEN index. Convert to a doc+rel index. 
        doc_idx, relative_idx = self.tokenidx_to_docidx(index)
        doc = self.dataset_csv.iloc[doc_idx]

        inp, trout, layer_names = self.get_model_trace(str(doc['decoded_prefix']), doc_idx)
        hidden_state = trout.squeeze()[relative_idx, :]

        # fixed bug here with the shuffling, didn't convert curr_token and target_token to one-hot. 
        onehots = np.eye(self.vocab_size)[inp.cpu()].squeeze()
        curr_token = torch.tensor(onehots[relative_idx], device='cpu')
        target_token = torch.tensor(onehots[relative_idx + self.target_idx], device='cpu')

        return torch.tensor(doc_idx, device='cpu'), hidden_state, curr_token, target_token 

'''
This collate function is meant to be used with the HiddenStateDataset.
'''
def hs_collate(batch):
    source_hss = []
    target_toks = []
    current_toks = []
    doc_idxs = []
    for hs in batch: 
        doc_idx, hidden_state, curr_token, target_token = (a.cpu() for a in hs)
        
        # save the hidden state and target token 
        source_hss.append(hidden_state.unsqueeze(0))
        target_toks.append(target_token.unsqueeze(0))
        current_toks.append(curr_token.unsqueeze(0))
        doc_idxs.append(doc_idx)

    source_hss = torch.cat(source_hss)
    target_toks = torch.cat(target_toks)
    current_toks = torch.cat(current_toks)
    doc_idxs = torch.tensor(doc_idxs, device='cpu')

    return (source_hss, target_toks, current_toks, doc_idxs)

'''
Dataset that retrieves hidden representations for a document as a "slice" of a model,
i.e. for a set layer, retrieves all the hidden states for document tokens at that layer,
as well as all the tokens for that document. 

Labels are one-hot encodings of the CURRENT token (i.e. tokens[i] is the one-hot encoding
of the token that hidden_states[i] encodes). For train_probe.py, target_idx handling
happens in the collate function when we're loading the data.
'''
class DocDataset(Dataset):
    def __init__(self, model, tokenizer, model_name, layer_name, target_idx, dataset_csv, window_size, vocab_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.layer_name = layer_name # int: -1 is embedding, 0-27 for layers, 28 for logits right at the end
        self.target_idx = target_idx # -1 is previous token, 0 is current. 
        self.dataset_csv = dataset_csv
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.device = device

        if self.target_idx not in [0, -1]:
            raise Exception(f"target_idx {self.target_idx} not yet supported (doc_collate will not work).")
    
    # when tokenizing make sure to include BOS token at beginning as "padding"
    def tokenize(self, text):
        if "llama" in self.model_name:
            t = torch.tensor(self.tokenizer.encode(text, bos=True, eos=False))
        else:
            bos = self.tokenizer.bos_token
            t = self.tokenizer(bos+text)
        
        if len(t) > self.window_size:
            return t[:self.window_size]
        else:
            return t
    
    def get_model_trace(self, text):
        if "llama" in self.model_name:
            tokenized = self.tokenize(text)
            inp = tokenized.unsqueeze(0).to(self.device)
            layers = [n for n, _ in self.model.named_modules() if re.match(r'^tok_embeddings|^layers.\d+$|^norm', n)]
            with baukit.Trace(self.model, layers[self.layer_name+1]) as tr:
                _ = self.model(inp, 0) # (bsz=1, doc_length, vocab_size)
            return inp, tr.output, layers
        
        else:
            inp = {k: torch.tensor(v)[None].to(self.device) for k, v in self.tokenize(text).items()}
            layers = [n for n, _ in self.model.named_modules() if re.match(r'^transformer.wte|^transformer.h.\d+$|^ln_f', n)]
            with baukit.Trace(self.model, layers[self.layer_name+1]) as tr:
                _ = self.model(**inp)
            inp = inp['input_ids']
            return inp, tr.output[0], layers
    
    def __len__(self):
        return len(self.dataset_csv) # number of docs, not tokens

    def __getitem__(self, index):
        doc = self.dataset_csv.iloc[index]
        inp, trout, layer_names = self.get_model_trace(str(doc['decoded_prefix']))
        hidden_states = trout.squeeze() # (doc_length, model_dim) 

        # turn token idxs into one-hot encodings
        targets = torch.tensor(np.eye(self.vocab_size)[inp.cpu()], device='cpu').squeeze()

        assert len(hidden_states) == len(targets)
        return torch.tensor(index), hidden_states, targets


'''
This collate function is meant to be used with the DocDataset. Has to be a class
so that target_idx can be specified. 

Collate function for predicting the previous token from a current hidden state.
Flattens a batch of several documents' hidden states w/ their corresponding tokens
into single arrays of hidden_states, previous_tokens, current_tokens, and doc_idxs
(so that document information can be recovered during testing). 

"num_batch_tokens" refers to the sum of the length of each document in the given 
batch. each output array is of length num_batch_tokens.

Parameters:
    batch: list of tuples, where each tuple represents a document. for example, 
        (0, hidden_states, tokens) contains the hidden states and corresponding
        tokens for document 0 in the dataset. document index 0 indicates the first
        document in the entire dataset, not just in that batch.

Returns:
    source_hss: source hidden states, shape (num_batch_tokens, model_dim)
    target_toks: one-hot previous token embeddings, shape (num_batch_tokens, vocab_size)
    current_toks: one-hot current token embeddings, shape (num_batch_tokens, vocab_size)
    doc_idxs: document indices each hidden state came from, shape (num_batch_tokens,)
'''
class DocCollate(object):
    def __init__(self, target_idx):
        self.target_idx = target_idx

        if self.target_idx not in [0, -1]:
            raise Exception(f"target_idx {self.target_idx} not yet supported (DocCollate will not work).")

    def __call__(self, batch):
        source_hss = []
        target_toks = []
        current_toks = []
        doc_idxs = []
        for doc in batch: 
            # batch looks like [doc0:(0, hidden_states, tokens), doc1:(1, hidden_states, tokens)...]
            doc_idx, hidden_states, tokens = (a.cpu() for a in doc)

            # hardcoded - take all of the hidden states after BOS token, and all of the token labels except for the last one 
            # source_hss: BOS [this is an example sentence]
            # target_toks: [BOS this is an example] sentence
            if self.target_idx == -1:
                source_hss.append(hidden_states[1:])
                target_toks.append(tokens[:-1])
                current_toks.append(tokens[1:])
                doc_idxs.append(torch.tensor([doc_idx for _ in range(len(hidden_states[1:]))], device='cpu'))
            
            # exclude predicting bos_embedding -> BOS
            elif self.target_idx == 0:
                source_hss.append(hidden_states[1:])
                target_toks.append(tokens[1:])
                current_toks.append(tokens[1:])
                doc_idxs.append(torch.tensor([doc_idx for _ in range(len(hidden_states[1:]))], device='cpu'))

        source_hss = torch.cat(source_hss)
        target_toks = torch.cat(target_toks)
        current_toks = torch.cat(current_toks)
        doc_idxs = torch.cat(doc_idxs)

        return (source_hss, target_toks, current_toks, doc_idxs)

