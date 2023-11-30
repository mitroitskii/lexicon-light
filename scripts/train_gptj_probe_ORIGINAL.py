'''
Running this file will train a linear probe to predict the current (0) or 
previous (-1) token at a specific layer: tok_embeddings, layer.[0-31], output. 
'''
import sys
import gc
import torch
import argparse
import os
import csv
import wandb
import baukit

import numpy as np
import pandas as pd
import torch.nn.functional as F

from transformers import GPTJForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_warmup as warmup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# adding a parent directory to the path so that we can import from modules

from modules.training import EarlyStopper, LinearModel, _topkprobs, _topktoks, weighted_mse
from modules.state_data import HiddenStateDataset, hs_collate, DocDataset, DocCollate

gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()

WINDOW_SIZE = 2048
MODEL_NAME = "EleutherAI/gpt-j-6b" #"lmsys/vicuna-7b-v1.3"
VOCAB_SIZE = 50400
model, tokenizer = (
    GPTJForCausalLM.from_pretrained(MODEL_NAME, revision="float16", torch_dtype=torch.float16).cuda(),
    AutoTokenizer.from_pretrained(MODEL_NAME) 
)
baukit.set_requires_grad(False, model)

torch.set_default_dtype(torch.float32)

# ??? why use logilens?
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

    for batch_idx, (data, target, currs, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        assert(not torch.isnan(data).any() and not torch.isinf(data).any())
        assert(not torch.isnan(target).any() and not torch.isinf(target).any())

        with torch.no_grad():
            curr_embeddings = model.transformer.wte(_topktoks(currs).to(device)).squeeze()
            if len(curr_embeddings.shape) == 1: # undo too much squeezing
                curr_embeddings = curr_embeddings.unsqueeze(0)
            inputs = torch.cat([data, curr_embeddings], dim=1) # (bsz, 4096) + (bsz, 4096) = (bsz, 8192)

        output = probe(inputs.float()).to(device)
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
            train_acc = 100 * (sum(_topktoks(output) == _topktoks(target)) / len(output))
            if torch.isinf(train_acc).any(): 
                print('denom', len(output), 'numerator', sum(_topktoks(output) == _topktoks(target)))
                print(_topktoks(output), _topktoks(target))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining Acc:{:3.3f}%\tBatch Loss: {:.6f} ({} tokens)'.format(
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), 
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
        for (data, targets, currs, doc_idxs) in baukit.pbar(test_loader):
            data, targets = data.to(device), targets.to(device)
            with torch.no_grad():
                curr_embeddings = model.transformer.wte(_topktoks(currs).to(device)).squeeze()
                if len(curr_embeddings.shape) == 1: # undo too much squeezing
                    curr_embeddings = curr_embeddings.unsqueeze(0)
                inputs = torch.cat([data, curr_embeddings], dim=1) # (bsz, 4096) + (bsz, 4096) = (bsz, 8192)

            output = probe(inputs.float()).to(device)

            loss = criterion(output, targets, reduction="mean")
            total_loss += loss.detach().item()

            # save all the pred vs target, along with the current token.
            for i, v in enumerate(output):         
                doc_id = doc_idxs[i]
                current_tok = _topktoks(currs[i])
                actual_tok = _topktoks(targets[i]) # target[i] is one-hot vector 
                predicted_tok = _topktoks(v)

                total_toks += 1
                if predicted_tok == actual_tok:
                    correct += 1
                if actual_tok in _topktoks(v, k=5):
                    topk_correct += 1
                
                if return_results:
                    # "what is the source hidden state encoding rn in terms of output?"
                    # probably useless but interesting to see just in case 
                    sourcehs_logitlens = _topktoks(logitlens(data[i]))

                    # BOS token becomes encoded as NaN in pandas here
                    curr_result = {
                        "doc_id" : doc_id.item(),
                        "current_tok_id" : current_tok.item(),
                        "actual_tok_id" : actual_tok.item(),
                        "predicted_tok_id" : predicted_tok.item(),
                        "current_tok" : tokenizer.decode(current_tok.tolist()),
                        "actual_tok" : tokenizer.decode(actual_tok.tolist()),
                        "predicted_tok" : tokenizer.decode(predicted_tok.tolist()),
                        "sourcehs_logitlens_tok_id" : sourcehs_logitlens.item(),
                        "sourcehs_logitlens_tok" : tokenizer.decode(sourcehs_logitlens.tolist()),
                        **_topkprobs(v, tokenizer)
                    }

                    results.append(curr_result)
                 
    test_loss = total_loss / len(test_loader) # divide total average loss by no. batches
    test_acc = 100 * correct / total_toks
    topk_acc = 100 * topk_correct / total_toks
    
    if return_results:
        return test_loss, test_acc, topk_acc, pd.DataFrame(results)
    else:
        return test_loss, test_acc, topk_acc


def main(args):
    run_name = add_args(f"LAYER{args.layer}-TGTIDX{args.target_idx}-{datasetname(args.train_data)}", args)
    wandb.init(project = args.wandb_proj, name = run_name, config = args, settings=wandb.Settings(start_method="fork"))
 
    print("shuffle: HiddenStateDataset+hs_collate" if args.shuffle else "no-shuffle: DocDataset+doc_collate")
    # if args.shuffle:
    #     raise Exception("Shuffling not yet implemented for GPT-J hidden states. Please use --no-shuffle.")
    #     sys.exit()
    if not args.shuffle and args.probe_bsz > 10:
        print(f"Warning: when --not-shuffle, batch size represents number of documents. {args.probe_bsz}>10, you may want to use a smaller batch size.")

    # make dirs that include the wandb id
    checkpoint_dir = f"../checkpoints/{MODEL_NAME}/{run_name}-{wandb.run.id}"
    log_dir = f"../logs/{MODEL_NAME}/{run_name}-{wandb.run.id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_data = pd.read_csv(args.train_data)
    val_data = pd.read_csv(args.val_data)
    test_data = pd.read_csv(args.test_data)
    
    # # check if previous runs have cached the needed hidden states. if not, generate and cache.
    # train_cache_dir = f"../data/hs_cache/{MODEL_NAME}/{datasetname(args.train_data)}"
    # val_cache_dir = f"../data/hs_cache/{MODEL_NAME}/{datasetname(args.val_data)}"
    # test_cache_dir = f"../data/hs_cache/{MODEL_NAME}/{datasetname(args.test_data)}"


    collate_fn = hs_collate if args.shuffle else DocCollate(args.target_idx)
    dset = HiddenStateDataset if args.shuffle else DocDataset
    train_dataset = dset(model, tokenizer, MODEL_NAME, args.layer, args.target_idx, train_data, WINDOW_SIZE, VOCAB_SIZE, device)
    val_dataset = dset(model, tokenizer, MODEL_NAME, args.layer, args.target_idx, val_data, WINDOW_SIZE, VOCAB_SIZE, device)
    test_dataset = dset(model, tokenizer, MODEL_NAME, args.layer, args.target_idx, test_data, WINDOW_SIZE, VOCAB_SIZE, device)

    # initialize matrix with inverted embed matrix if desired. have to stack it along with zeros 
    init_matrix = None 
    if args.probe_init == 'pinv':
        embed_weights = model.transformer.wte.weight # model.tok_embeddings.weight if "llama" in MODEL_NAME else
        inverse = torch.tensor(np.linalg.pinv(np.array(embed_weights.cpu(), dtype=np.float32)))
        init_matrix = torch.cat([inverse, torch.zeros_like(inverse)], dim=0).transpose(0, 1) # creates (89192, 50400).T
        # (4096, 50400) inverse, a wide rectangle
        # (8192, 50400) two wide rectangles stacked on top of e/o
        # so we want to make that second rectangle just zeros. 
    
    linear_probe = LinearModel(model.lm_head.in_features * 2, VOCAB_SIZE, init_matrix=init_matrix).to(device)
    wandb.watch(linear_probe)

    # drop_last must be True for val and test so that we can average over batch loss avgs. 
    kwargs = {}
    if args.num_workers > 0:
        kwargs['num_workers'] = args.num_workers
        kwargs['multiprocessing_context'] = 'spawn' # helps with lab machines
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.probe_bsz, collate_fn=collate_fn, 
        drop_last=True, pin_memory=True, shuffle=args.shuffle, **kwargs) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.probe_bsz, collate_fn=collate_fn, 
        drop_last=True, pin_memory=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.probe_bsz, collate_fn=collate_fn, 
        drop_last=True, pin_memory=True, **kwargs)

    optimizer = None
    warmup_scheduler = None
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(linear_probe.parameters(), lr=args.probe_lr, momentum=args.probe_momentum, weight_decay=args.probe_wd)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_wd) # no momentum
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    
    criterion = {
        "weighted_mse" : weighted_mse,
        "ce" : F.cross_entropy,
        "mse" : F.mse_loss
    }[args.criterion]
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)

    print('training linear probe...') 
    batches_seen = 0
    for epoch in range(args.probe_epochs):
        print('# Epoch {} #'.format(epoch))
        batches_seen = train_epoch(epoch, linear_probe, train_loader, criterion, optimizer, warmup_scheduler, args.accumulate, args.clip_threshold, batches_seen)

        # log validation loss at the end of each epoch to decide early stopping
        val_loss, val_acc, val_topk_acc = test(linear_probe, val_loader, criterion, return_results=False)
        wandb.log({"val_loss": val_loss, "val_acc": val_acc, "val_topk_acc": val_topk_acc})

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                scheduler.step(val_loss)
        else:
            scheduler.step(val_loss)

        if early_stopper.early_stop(val_loss):
            print(f"early stopping after Epoch {epoch}, val_loss={val_loss} min_val_loss={early_stopper.min_val_loss}")
            break
    
    # Get final testing accuracy and prediction results
    torch.save(linear_probe.state_dict(), f"{checkpoint_dir}/final.ckpt")
    test_loss, test_acc, test_topk_acc, test_results = test(linear_probe, test_loader, criterion, return_results=True)
    test_results.to_csv(log_dir + f"/{datasetname(args.test_data)}_results.csv", quoting=csv.QUOTE_ALL)
    
    print('Test Loss: {:10.4f}  Accuracy: {:3.4f}%\n'.format(test_loss, test_acc))
    wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_topk_acc": test_topk_acc})
    wandb.finish()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training info for linear probe
    parser.add_argument('--probe_bsz', type=int, default=1024)
    parser.add_argument('--probe_lr', type=float, default=1.2)
    parser.add_argument('--probe_wd', type=float, default=1e-3)
    parser.add_argument('--probe_momentum', type=float, default=1.6)
    parser.add_argument('--probe_epochs', type=int, default=3)
    parser.add_argument('--probe_init', type=str, choices=['random', 'pinv'], default='random',
                        help="whether to initialize probe `random` or using the pseudo-inverse of the embed matrix `pinv`.")
    
    parser.add_argument('--wandb_proj', type=str, default='lexicon-cat-probes')
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--clip_threshold', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW'], default='AdamW')
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--criterion', type=str, choices=['weighted_mse', 'ce', 'mse'], default='ce')

    # document data locations
    parser.add_argument('--train_data', type=str, default='../data/train_tiny_1000.csv')
    parser.add_argument('--val_data', type=str, default='../data/val_tiny_500.csv')
    parser.add_argument('--test_data', type=str, default='../data/test_tiny_500.csv')

    # required specifications for where probe is trained 
    parser.add_argument('--layer', type=int, required=True,
                        help='which layer to train the probe at. from -1..28 where -1 is embedding layer and 28 is output.')
    parser.add_argument('--target_idx', type=int, required=True, # help msg uses NEW numbering
                        help='which token probe should predict from current hidden state (e.g. 0 for current token, -1 for prev)')

    args = parser.parse_args()
    main(args)
    
    
