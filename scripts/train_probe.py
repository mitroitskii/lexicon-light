'''
Running this file will train a probe to predict the current (0) or 
previous (-1) token at a specific layer: tok_embeddings, layer.[0-31], output. 
'''
# partially adapted from https://github.com/sfeucht/lexicon and modified as part of a research project at David Bau's lab

# to run this script, use the the `run_probe.sh` script from the `scripts/` folder
# the run takes ~3GB of GPU memory per batch, so you may need to reduce the batch size if you run out of memory (see the arguments at the bottom of the script)
# to be able to log the run with wandb, you need to have a wandb account and be logged in via a wandb cli tool (https://docs.wandb.ai/quickstart)

import sys
import gc
import torch
import argparse
import os
import wandb
import baukit # useful utitilies for interpetability
import pandas as pd
import torch.nn.functional as F

from transformers import GPTJForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_warmup as warmup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# adding a parent directory to the path so that we can import from modules (this line of code needs to be here in the middle before the modules are imported)

from modules.training import _topkprobs, _topktoks
from modules.models import LinearModel, MLPModel, RNNModel
from modules.state_data import DocDataset, DocCollate

gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()

WINDOW_SIZE = 2048 # truncate documents from training data to this many tokens
MODEL_NAME = "EleutherAI/gpt-j-6b"
VOCAB_SIZE = 50400
model, tokenizer = (
    GPTJForCausalLM.from_pretrained(MODEL_NAME, revision="float16", torch_dtype=torch.float16).cuda(),
    AutoTokenizer.from_pretrained(MODEL_NAME) 
)
baukit.set_requires_grad(False, model) # freeze the model

torch.set_default_dtype(torch.float32)

def datasetname(input):
    # returns the name of the dataset from the path
    return input.split('/')[-1][:-4]

def add_args(s, args):
    # adds the arguments to the run name
    for k, v in vars(args).items():
        if k in ['probe_bsz', 'probe_epochs']:
            s += f"-{k[6:]}{v}"
        elif k in ['probe_lr']:
            s += f"-{k[6:]}" + "{:1.5f}".format(v)
    return s

def train_epoch(epoch, probe, train_loader, criterion, optimizer, warmup_scheduler, accumulate, clip_threshold, batches_seen):
    '''
    Trains a probe for a single epoch and logs loss/accuracy.

    Parameters:
        epoch: index of the current epoch
        probe: model to be trained
        train_loader: DataLoader of training data
        criterion: loss function
        optimizer: torch optimizer for the model
        warmup_scheduler: learning rate warmup scheduler
        accumulate: number of batches to accumulate gradients over
        clip_threshold: threshold for gradient clipping
        batches_seen: number of batches seen so far

    Returns:
        None
    '''
    probe.train()

    for batch_idx, (curr_hss, targets, curr_tokens, _) in enumerate(train_loader):
        curr_hss, targets = curr_hss.to(device), targets.to(device).float()
        # check for NaNs/Infs
        assert(not torch.isnan(curr_hss).any() and not torch.isinf(curr_hss).any()) 
        assert(not torch.isnan(targets).any() and not torch.isinf(targets).any())

        with torch.no_grad():
            curr_embeddings = model.transformer.wte(_topktoks(curr_tokens).to(device)).squeeze() # wte is the embedding layer
            if len(curr_embeddings.shape) == 1: # undo too much squeezing
                curr_embeddings = curr_embeddings.unsqueeze(0)
            inputs = torch.cat([curr_hss, curr_embeddings], dim=1) # (total_batch_tokens, 4096) + (total_batch_tokens, 4096) = (total_batch_tokens, 8192); 4096 is the size of model layers

        output = probe(inputs.float()).to(device)
        loss = criterion(output, targets, reduction="mean").float()
        loss.backward() 
        
        if batch_idx % accumulate == 0 and batch_idx > 0: # accumulate gradients over accumulate batches
            torch.nn.utils.clip_grad_norm_(probe.parameters(), clip_threshold)
            optimizer.step()
            optimizer.zero_grad()

        loss = loss.detach().item() # detach loss from the graph

        # learning rate warmup for AdamW. 
        if warmup_scheduler is not None:
            if batch_idx < len(train_loader)-1:
                with warmup_scheduler.dampening():
                    pass

        # print training accuracy/loss every 10 epochs, and on the last epoch
        if batch_idx % max(accumulate, 10) == 0 or batch_idx == len(train_loader) - 1:
            train_acc = 100 * (sum(_topktoks(output) == _topktoks(targets)) / len(output))
            if torch.isinf(train_acc).any(): 
                print('denom', len(output), 'numerator', sum(_topktoks(output) == _topktoks(targets)))
                print(_topktoks(output), _topktoks(targets)) # print the tokens that are causing the inf
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining Acc:{:3.3f}%\tBatch Loss: {:.6f} ({} tokens)'.format(
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), 
                train_acc.item(), loss, curr_hss.size()[0]))

            wandb.log({"train_loss": loss, "train_acc": train_acc, 
                       "epoch": epoch, "batches_seen": 1 + batch_idx + batches_seen})

    return 1 + batch_idx + batches_seen # return total number of batches seen so far

def test(probe, test_loader, criterion, k=5):
    '''
    Tests a probe on the given data.

    Parameters:
        probe: a model to test
        test_loader: DataLoader containing testing data
        criterion: loss function
        k: top k accuracy
    Returns:
        test_loss: average test loss for data in test_loader
        test_acc: average test accuracy for data in test_loader
    '''
    probe.eval()
    total_loss = 0.0
    total_toks = 0
    correct = 0
    topk_correct = 0
    with torch.no_grad():
        for (curr_hss, targets, curr_tokens, doc_idxs) in baukit.pbar(test_loader):
            curr_hss, targets = curr_hss.to(device), targets.to(device)
            with torch.no_grad():
                curr_embeddings = model.transformer.wte(_topktoks(curr_tokens).to(device)).squeeze()
                if len(curr_embeddings.shape) == 1: # undo too much squeezing
                    curr_embeddings = curr_embeddings.unsqueeze(0)
                inputs = torch.cat([curr_hss, curr_embeddings], dim=1) # (total_batch_tokens, 4096) + (total_batch_tokens, 4096) = (total_batch_tokens, 8192)

            output = probe(inputs.float()).to(device)

            loss = criterion(output, targets, reduction="mean")
            total_loss += loss.detach().item()

            # save all the pred vs target, along with the current token.
            for i, v in enumerate(output):         
                actual_tok = _topktoks(targets[i]) # target[i] is one-hot vector 
                predicted_tok = _topktoks(v)

                total_toks += 1
                if predicted_tok == actual_tok:
                    correct += 1
                if actual_tok in _topktoks(v, k=5):
                    topk_correct += 1
                 
    test_loss = total_loss / len(test_loader) # divide total average loss by no. batches
    test_acc = 100 * correct / total_toks
    topk_acc = 100 * topk_correct / total_toks
    
    return test_loss, test_acc, topk_acc

def main(args):
    """
    Main function for training a probe model.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """

    run_name = add_args(f"LAYER{args.layer}-TGTIDX{args.target_idx}-{datasetname(args.train_data)}", args)
    wandb.init(project = args.wandb_proj, name = run_name, config = args, settings=wandb.Settings(start_method="fork"))
 
    if args.probe_bsz > 10:
        print(f"The batch size represents number of documents. {args.probe_bsz}>10, you may want to use a smaller batch size.")

    train_data = pd.read_csv(args.train_data)
    val_data = pd.read_csv(args.val_data)
    test_data = pd.read_csv(args.test_data)

    collate_fn = DocCollate(args.target_idx)
    dset = DocDataset
    train_dataset = dset(model, tokenizer, MODEL_NAME, args.layer, args.target_idx, train_data, WINDOW_SIZE, VOCAB_SIZE, device)
    val_dataset = dset(model, tokenizer, MODEL_NAME, args.layer, args.target_idx, val_data, WINDOW_SIZE, VOCAB_SIZE, device)
    test_dataset = dset(model, tokenizer, MODEL_NAME, args.layer, args.target_idx, test_data, WINDOW_SIZE, VOCAB_SIZE, device)
    
    probe = None
    if args.probe_model == 'linear':
        probe = LinearModel(model.lm_head.in_features * 2, VOCAB_SIZE).to(device) # lm_head.in_features is the size of the model layers; *2 because we concatenate the current hidden state with the current token embedding
        wandb.watch(probe)
    elif args.probe_model == 'mlp':
        probe = MLPModel(model.lm_head.in_features * 2, VOCAB_SIZE, dropout_rate=args.probe_dropout).to(device)
        wandb.watch(probe)
    elif args.probe_model == 'rnn':
        probe = RNNModel(model.lm_head.in_features * 2, VOCAB_SIZE).to(device)
        wandb.watch(probe)

    # drop_last determines what happens when the total number of samples in a dataset is not a multiple of the batch size.
    # this must be True for val and test so that we can average over batch loss avgs
    kwargs = {}
    if args.num_workers > 0:
        kwargs['num_workers'] = args.num_workers
        kwargs['multiprocessing_context'] = 'spawn' # helps with lab machines
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.probe_bsz, collate_fn=collate_fn, 
        drop_last=True, pin_memory=True, shuffle=False, **kwargs) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.probe_bsz, collate_fn=collate_fn, 
        drop_last=True, pin_memory=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.probe_bsz, collate_fn=collate_fn, 
        drop_last=True, pin_memory=True, **kwargs)

    optimizer = None
    warmup_scheduler = None
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(probe.parameters(), lr=args.probe_lr, momentum=args.probe_momentum, weight_decay=args.probe_wd)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_wd) # no momentum
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    
    criterion = F.cross_entropy # TODO might want to use BCELoss for RNNModel
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # print which model is trained depending on the args
    print(f"Training {args.probe_model} probe at layer {args.layer}")

    batches_seen = 0
    for epoch in range(args.probe_epochs):
        print('# Epoch {} #'.format(epoch))
        batches_seen = train_epoch(epoch, probe, train_loader, criterion, optimizer, warmup_scheduler, args.accumulate, args.clip_threshold, batches_seen)

        val_loss, val_acc, val_topk_acc = test(probe, val_loader, criterion)
        wandb.log({"val_loss": val_loss, "val_acc": val_acc, "val_topk_acc": val_topk_acc})

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                scheduler.step(val_loss)
        else:
            scheduler.step(val_loss)
    
    # Get final testing accuracy
    test_loss, test_acc, test_topk_acc = test(probe, test_loader, criterion)
    
    print('Test Loss: {:10.4f}  Accuracy: {:3.4f}%\n'.format(test_loss, test_acc))
    wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_topk_acc": test_topk_acc})
    wandb.finish()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training info for the probe
    parser.add_argument('--probe_model', type=str, choices=['linear', 'mlp', 'rnn'], default='linear')
    parser.add_argument('--probe_bsz', type=int, default=6)
    parser.add_argument('--probe_lr', type=float, default=2.18)
    parser.add_argument('--probe_wd', type=float, default=0.003)
    parser.add_argument('--probe_dropout', type=float, default=0.5)
    parser.add_argument('--probe_momentum', type=float, default=1.53)
    parser.add_argument('--probe_epochs', type=int, default=14)    
    parser.add_argument('--wandb_proj', type=str, default='lexicon-cat-probes')
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--clip_threshold', type=float, default=0.16)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW'], default='AdamW')
    parser.add_argument('--num_workers', type=int, default=0)

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
