'''
Running this file will train a linear probe to predict the current (0) or 
token from the embedding layer for GPT-J, but training it on an evenly-sampled
distribution of all of the embeddings across the vocab.
'''
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
# adding a parent directory to the path so that we can import from modules @dmitrii

from modules.state_data import hs_collate
from modules.training import EarlyStopper, LinearModel, _topkprobs, _topktoks, weighted_mse, svd_init

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
        if "llama" in self.model_name:
            t = torch.tensor(self.tokenizer.encode(text, bos=True, eos=False))
        else:
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
        # inp = torch.tensor(index).to(self.device)

        # if "llama" in self.model_name:
        #     embed = model.tok_embeddings(inp).squeeze().cpu().float()
        # else:
        #     embed = model.transformer.wte(inp).squeeze().cpu().float()

        # onehot = torch.tensor(np.eye(self.vocab_size)[index], device='cpu')
        # return torch.tensor(0, device='cpu'), embed, onehot, onehot

# the workhorse.


def allembeds_collate(batch):
    # batch is a list of indexes
    inp = torch.tensor(batch).to(device)

    if "llama" in MODEL_NAME:
        embed = model.tok_embeddings(inp).squeeze().cpu().float()
    else:
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
            if torch.isinf(train_acc).any():
                print('denom', len(output), 'numerator', sum(
                    _topktoks(output) == _topktoks(target)))
                print(_topktoks(output), _topktoks(target))
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
    return_results: boolean indicating whether to 

Returns:
    test_loss: average test loss for data in test_loader
    test_acc: average test accuracy for data in test_loader
    results: if return_results, pandas DataFrame containing predicted tokens, current tokens, etc.
'''


def test(probe, test_loader, criterion, return_results=False, k=5):
    probe.eval()
    total_loss = 0.0
    total_toks = 0
    correct = 0
    topk_correct = 0
    results = []
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

                if return_results:

                    # BOS token becomes encoded as NaN in pandas here
                    curr_result = {
                        "actual_tok_id": actual_tok.item(),
                        "predicted_tok_id": predicted_tok.item(),
                        "actual_tok": tokenizer.decode(actual_tok.tolist()),
                        "predicted_tok": tokenizer.decode(predicted_tok.tolist()),
                        **_topkprobs(v, tokenizer)
                    }

                    results.append(curr_result)

    # divide total average loss by no. batches
    test_loss = total_loss / len(test_loader)
    test_acc = 100 * correct / total_toks
    topk_acc = 100 * topk_correct / total_toks

    if return_results:
        return test_loss, test_acc, topk_acc, pd.DataFrame(results)
    else:
        return test_loss, test_acc, topk_acc


def main(args):
    run_name = add_args("SANITYPROBE", args)
    wandb.init(project=args.wandb_proj, name=run_name, config=args,
               settings=wandb.Settings(start_method="fork"))
    print("shuffle: HiddenStateDataset+hs_collate")

    # make dirs that include the wandb id
    checkpoint_dir = f"../checkpoints/{MODEL_NAME}/{run_name}-{wandb.run.id}"
    log_dir = f"../logs/{MODEL_NAME}/{run_name}-{wandb.run.id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = AllEmbeds(model, tokenizer, MODEL_NAME,
                        WINDOW_SIZE, VOCAB_SIZE, device)

    # initialize matrix with inverted embed matrix if desired. have to stack it along with zeros
    init_matrix = None
    if args.probe_init == 'pinv':
        # model.tok_embeddings.weight if "llama" in MODEL_NAME else
        embed_weights = model.transformer.wte.weight
        inverse = torch.tensor(np.linalg.pinv(
            np.array(embed_weights.cpu(), dtype=np.float32)))

        # for when you're not catting
        init_matrix = inverse.transpose(0, 1)
        # for when you're catting
        # init_matrix = torch.cat([inverse, torch.zeros_like(inverse)], dim=0).transpose(0, 1) # creates (89192, 50400).T
        # (4096, 50400) inverse, a wide rectangle
        # (8192, 50400) two wide rectangles stacked on top of e/o
        # so we want to make that second rectangle just zeros.

    linear_probe = LinearModel(
        model.lm_head.in_features, VOCAB_SIZE, init_matrix=init_matrix).to(device)
    wandb.watch(linear_probe)

    # drop_last must be True for val and test so that we can average over batch loss avgs.
    kwargs = {}
    # if args.num_workers > 0:
    #     kwargs['num_workers'] = args.num_workers
    #     kwargs['multiprocessing_context'] = 'spawn' # helps with lab machines, but not on cais
    train_loader = DataLoader(dataset=dataset, batch_size=args.probe_bsz, collate_fn=allembeds_collate,
                              drop_last=False, pin_memory=True, shuffle=True, **kwargs)
    val_loader = DataLoader(dataset=dataset, batch_size=args.probe_bsz, collate_fn=allembeds_collate,
                            drop_last=False, pin_memory=True, **kwargs)

    if args.probe_init == 'leastsq':
        temp_loader = DataLoader(dataset=dataset, batch_size=len(
            dataset), collate_fn=allembeds_collate, drop_last=False, pin_memory=True, shuffle=False, **kwargs)
        (data, targets) = next(iter(temp_loader))
        svd_init(linear_probe, data, targets)

    optimizer = None
    warmup_scheduler = None
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(linear_probe.parameters(
        ), lr=args.probe_lr, momentum=args.probe_momentum)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            linear_probe.parameters(), lr=args.probe_lr)  # no momentum
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    criterion = {
        "weighted_mse": weighted_mse,
        "ce": F.cross_entropy,
        "mse": F.mse_loss
    }[args.criterion]
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)

    print('training linear probe...')
    batches_seen = 0
    for epoch in range(args.probe_epochs):
        print('# Epoch {} #'.format(epoch))
        batches_seen = train_epoch(epoch, linear_probe, train_loader, criterion, optimizer,
                                   warmup_scheduler, args.accumulate, args.clip_threshold, batches_seen)

        # log validation loss at the end of each epoch to decide early stopping
        val_loss, val_acc, val_topk_acc = test(
            linear_probe, val_loader, criterion, return_results=False)
        wandb.log({"val_loss": val_loss, "val_acc": val_acc,
                  "val_topk_acc": val_topk_acc})

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                scheduler.step(val_loss)
        else:
            scheduler.step(val_loss)

        if early_stopper.early_stop(val_loss):
            print(
                f"early stopping after Epoch {epoch}, val_loss={val_loss} min_val_loss={early_stopper.min_val_loss}")
            break

    # Get final testing accuracy and prediction results
    torch.save(linear_probe.state_dict(), f"{checkpoint_dir}/final.ckpt")
    test_loss, test_acc, test_topk_acc, test_results = test(
        linear_probe, val_loader, criterion, return_results=True)
    test_results.to_csv(
        log_dir + f"/sanityprobe_results.csv", quoting=csv.QUOTE_ALL)

    print('Test Loss: {:10.4f}  Accuracy: {:3.4f}%\n'.format(
        test_loss, test_acc))
    wandb.log({"test_loss": test_loss, "test_acc": test_acc,
              "test_topk_acc": test_topk_acc})
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training info for linear probe
    parser.add_argument('--probe_bsz', type=int, default=5)
    parser.add_argument('--probe_lr', type=float, default=5e-5)
    parser.add_argument('--probe_momentum', type=float, default=0.9)
    parser.add_argument('--probe_epochs', type=int, default=5)
    parser.add_argument('--probe_init', type=str, choices=['random', 'pinv', 'leastsq'], default='random',
                        help="whether to initialize probe `random` or using the pseudo-inverse of the embed matrix `pinv`.")

    parser.add_argument('--wandb_proj', type=str, default='lexicon-cat-probes')
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--clip_threshold', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str,
                        choices=['SGD', 'AdamW'], default='SGD')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--criterion', type=str,
                        choices=['weighted_mse', 'ce', 'mse'], default='ce')

    args = parser.parse_args()
    main(args)
