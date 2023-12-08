'''
Train a model to predict a token from the embedding on an evenly-sampled
distribution of all of the embeddings across the vocabulary
'''
# partially adapted from https://github.com/sfeucht/lexicon and modified as part of a research project at David Bau's lab
# to run this script, use the following command:
# `python train_baseline.py`
# the run takes ~12GB-20 of GPU memory depending on the model, so make sure you have enough memory (linear is the smallest, then MLP, then RNN)
# to be able to log the run with wandb, you need to have a wandb account and be logged in via a wandb cli tool (https://docs.wandb.ai/quickstart)
import sys
import gc
import torch
import argparse
import os
import csv
import wandb    
import baukit
import regex as re
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import GPTJForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_warmup as warmup
import lovely_tensors as lt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# adding a parent directory to the path so that we can import from modules (this line of code needs to be here in the middle before the modules are imported)

from modules.state_data import hs_collate
from modules.training import _topkprobs, _topktoks
from modules.models import LinearModel, MLPModel, RNNModel

lt.monkey_patch()
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()

WINDOW_SIZE = 2048
MODEL_NAME = "EleutherAI/gpt-j-6b"  # "lmsys/vicuna-7b-v1.3"
VOCAB_SIZE = 50400
model, tokenizer = (
    GPTJForCausalLM.from_pretrained(
        MODEL_NAME, revision="float16", torch_dtype=torch.float16).cuda(),
    AutoTokenizer.from_pretrained(MODEL_NAME)
)
baukit.set_requires_grad(False, model)

torch.set_default_dtype(torch.float32)


class AllEmbeds(torch.utils.data.Dataset):
    def __init__(self, model, tokenizer, model_name, window_size, vocab_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.device = device

        self.layer_name = -1
        self.target_idx = 0

    # when tokenizing make sure to include BOS token at beginning as "padding"
    def tokenize(self, text):
        bos = self.tokenizer.bos_token
        t = self.tokenizer(bos+text)

        if len(t) > self.window_size:
            return t[:self.window_size]
        else:
            return t

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, index):
        return index

# the workhorse.


def allembeds_collate(batch):
    # batch is a list of indexes
    inp = torch.tensor(batch).to(device)
    embed = model.transformer.wte(inp).squeeze().cpu().float()

    onehot = torch.tensor(np.eye(VOCAB_SIZE)[batch], device='cpu')
    return embed, onehot


def logitlens(state):
    return model.lm_head(state)


def datasetname(input):
    return input.split('/')[-1][:-4]


def add_args(s, args):
    for k, v in vars(args).items():
        if k in ['probe_bsz', 'probe_epochs']:
            s += f"-{k[6:]}{v}"
        elif k in ['probe_lr']:
            s += f"-{k[6:]}" + "{:1.5f}".format(v)
    return s


'''
Trains a probe for a single epoch and logs loss/accuracy.

Parameters:
    epoch: index of the current epoch
    probe: linear model to be trained
    train_loader: DataLoader of training data
    optimizer: torch optimizer for linear model

Returns:
    None
'''


def train_epoch(epoch, probe, train_loader, criterion, optimizer, warmup_scheduler, accumulate, clip_threshold, batches_seen):
    probe.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        assert (not torch.isnan(data).any() and not torch.isinf(data).any())
        assert (not torch.isnan(target).any()
                and not torch.isinf(target).any())

        output = probe(data.float()).to(device)
        loss = criterion(output, target, reduction="mean").float()
        loss.backward()

        if batch_idx % accumulate == 0 and batch_idx > 0:
            # print(probe.fc.weight.grad[0][:10])
            torch.nn.utils.clip_grad_norm_(probe.parameters(), clip_threshold)
            optimizer.step()
            optimizer.zero_grad()

        loss = loss.detach().item()

        # learning rate warmup for AdamW.
        if warmup_scheduler is not None:
            if batch_idx < len(train_loader)-1:
                with warmup_scheduler.dampening():
                    pass

        # print training accuracy/loss every 10 epochs, and on the last epoch
        if batch_idx % max(accumulate, 10) == 0 or batch_idx == len(train_loader) - 1:
            train_acc = 100 * \
                (sum(_topktoks(output) == _topktoks(target)) / len(output))
            if torch.isinf(train_acc).any(): # if there are any infinities in the training accuracy
                print('denom', len(output), 'numerator', sum(
                    _topktoks(output) == _topktoks(target))) # print the numerator and denominator
                print(_topktoks(output), _topktoks(target)) # print the offending tokens
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining Acc:{:3.3f}%\tBatch Loss: {:.6f} ({} tokens)'.format(
                epoch, batch_idx, len(train_loader), 100. *
                batch_idx / len(train_loader),
                train_acc.item(), loss, data.size()[0]))

            wandb.log({"train_loss": loss, "train_acc": train_acc,
                       "epoch": epoch, "batches_seen": 1 + batch_idx + batches_seen})

    return 1 + batch_idx + batches_seen


'''
Tests a probe on the given data.

Parameters:
    probe: linear model to test
    test_loader: DataLoader containing testing data

Returns:
    test_loss: average test loss for data in test_loader
    test_acc: average test accuracy for data in test_loader
'''


def test(probe, test_loader, criterion, k=5):
    probe.eval()
    total_loss = 0.0
    total_toks = 0
    correct = 0
    topk_correct = 0
    with torch.no_grad():
        for (data, targets) in baukit.pbar(test_loader):
            data, targets = data.to(device), targets.to(device)

            output = probe(data.float()).to(device)

            loss = criterion(output, targets, reduction="mean")
            total_loss += loss.detach().item()

            # save all the pred vs target, along with the current token.
            for i, v in enumerate(output):
                # target[i] is one-hot vector
                actual_tok = _topktoks(targets[i])
                predicted_tok = _topktoks(v)

                total_toks += 1
                if predicted_tok == actual_tok:
                    correct += 1
                if actual_tok in _topktoks(v, k=5):
                    topk_correct += 1

    # divide total average loss by no. batches
    test_loss = total_loss / len(test_loader)
    test_acc = 100 * correct / total_toks
    topk_acc = 100 * topk_correct / total_toks

    return test_loss, test_acc, topk_acc


def main(args):
    run_name = add_args("BASELINE", args)
    wandb.init(project=args.wandb_proj, name=run_name, config=args,
               settings=wandb.Settings(start_method="fork"))

    dataset = AllEmbeds(model, tokenizer, MODEL_NAME,
                        WINDOW_SIZE, VOCAB_SIZE, device)

    probe = None
    if args.probe_model == 'linear':
        probe = LinearModel(model.lm_head.in_features, VOCAB_SIZE).to(device)
        wandb.watch(probe)
    elif args.probe_model == 'mlp':
        probe = MLPModel(model.lm_head.in_features, VOCAB_SIZE, dropout_rate=args.probe_dropout).to(device)
        wandb.watch(probe)
    elif args.probe_model == 'rnn':
        probe = RNNModel(model.lm_head.in_features, VOCAB_SIZE).to(device)
        wandb.watch(probe)

    # drop_last must be True for val and test so that we can average over batch loss avgs.
    kwargs = {}
    # if args.num_workers > 0:
    #     kwargs['num_workers'] = args.num_workers
    #     kwargs['multiprocessing_context'] = 'spawn' # helps with lab machines, but not on cais
    train_loader = DataLoader(dataset=dataset, batch_size=args.probe_bsz, collate_fn=allembeds_collate,
                              drop_last=False, pin_memory=True, shuffle=True, **kwargs)
    val_loader = DataLoader(dataset=dataset, batch_size=args.probe_bsz, collate_fn=allembeds_collate,
                            drop_last=False, pin_memory=True, **kwargs)

    optimizer = None
    warmup_scheduler = None
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(probe.parameters(
        ), lr=args.probe_lr, momentum=args.probe_momentum)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            probe.parameters(), lr=args.probe_lr)  # no momentum
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    criterion = F.cross_entropy
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    print('training linear probe...')
    batches_seen = 0
    for epoch in range(args.probe_epochs):
        print('# Epoch {} #'.format(epoch))
        batches_seen = train_epoch(epoch, probe, train_loader, criterion, optimizer,
                                   warmup_scheduler, args.accumulate, args.clip_threshold, batches_seen)

        # log validation loss at the end of each epoch to decide early stopping
        val_loss, val_acc, val_topk_acc = test(
            probe, val_loader, criterion)
        wandb.log({"val_loss": val_loss, "val_acc": val_acc,
                  "val_topk_acc": val_topk_acc})

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                scheduler.step(val_loss)
        else:
            scheduler.step(val_loss)


    # Get final testing accuracy
    test_loss, test_acc, test_topk_acc = test(
        probe, val_loader, criterion)

    print('Test Loss: {:10.4f}  Accuracy: {:3.4f}%\n'.format(
        test_loss, test_acc))
    wandb.log({"test_loss": test_loss, "test_acc": test_acc,
              "test_topk_acc": test_topk_acc})
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training info for linear probe
    parser.add_argument('--probe_model', type=str, choices=['linear', 'mlp', 'rnn'], default='linear')
    parser.add_argument('--probe_bsz', type=int, default=5)
    parser.add_argument('--probe_lr', type=float, default=5e-5)
    parser.add_argument('--probe_momentum', type=float, default=0.9)
    parser.add_argument('--probe_epochs', type=int, default=5)
    parser.add_argument('--probe_dropout', type=float, default=0.5)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--clip_threshold', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str,
                        choices=['SGD', 'AdamW'], default='SGD')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--wandb_proj', type=str, default='lexicon-cat-probes')

    args = parser.parse_args()
    main(args)
