'''What is a baseline for how well we can recover the original token from a model's
embedding matrix? just to sanity check why my probes are doing so poorly at the -1 layer.
'''
import regex as re
import numpy as np
import pandas as pd
import baukit
import torch
import torch.nn.functional as F
from transformers import GPTJForCausalLM, AutoTokenizer
from llama import Llama
import lovely_tensors as lt

lt.monkey_patch()

device = torch.device('cuda')

# Get GPT-J embeddings
# print("loading model...")
# MODEL_NAME = "EleutherAI/gpt-j-6b"
# model, tokenizer = (
#     GPTJForCausalLM.from_pretrained(MODEL_NAME, revision="float16", torch_dtype=torch.float16).cuda(),
#     AutoTokenizer.from_pretrained(MODEL_NAME) 
# )
# baukit.set_requires_grad(False, model)
MODEL_NAME = "llama-2-7b"
WINDOW_SIZE = 4096
MODEL_BSZ = 6
llama = Llama.build(
    ckpt_dir=f"../llama/{MODEL_NAME}", # can also use shared model on cais
    tokenizer_path="../llama/tokenizer.model",
    max_seq_len=WINDOW_SIZE, # adjust if memory issues
    max_batch_size=MODEL_BSZ
)
baukit.set_requires_grad(False, llama.model)
model = llama.model
tokenizer = llama.tokenizer

# load in data 
print(f"{MODEL_NAME} model loaded. loading data...")
df = pd.read_csv("../data/test_tiny_500.csv")
text = list(df['decoded_prefix'])

def tokenize(text, tokenizer=tokenizer):
    if "llama" in MODEL_NAME:
        toks = torch.tensor(tokenizer.encode(text, bos=True, eos=False))
    else:
        bos = tokenizer.bos_token
        toks = tokenizer(bos+text)
    return toks

# tokenize and get ground truth
if "llama" in MODEL_NAME:
    docs = [tokenize(t) for t in text]
else:
    docs = [tokenize(t)['input_ids'] for t in text]
flat_toks = [t for doc in docs for t in doc]

tokens = torch.tensor(flat_toks)
if "llama" in MODEL_NAME:
    curr_embeddings = model.tok_embeddings(tokens.to(device)).squeeze().cpu().float()
else:
    curr_embeddings = model.transformer.wte(tokens.to(device)).squeeze().cpu().float()
print("curr_embeddings", curr_embeddings)

# Get the inverse of the embedding matrix, apply and calculate accuracy 
# wte is shape (32000, 4096). inverse is shape (4096, 32000)
embed_weights = model.tok_embeddings.weight if "llama" in MODEL_NAME else model.transformer.wte.weight
inverse = torch.tensor(np.linalg.pinv(np.array(embed_weights.cpu(), dtype=np.float32))).cpu()

predicted_tokens = torch.matmul(curr_embeddings, inverse).topk(k=1, dim=-1)[1].squeeze()

print("predicted_tokens", predicted_tokens)
print("reversal acc (%)", 100 * (sum(predicted_tokens==tokens.cpu()) / len(flat_toks)).item())
print("---")
print("inverse", inverse)
print("samples", predicted_tokens.cpu().numpy()[:10], tokens.cpu().numpy()[:10])

allonehot = torch.tensor(np.arange(32000)).int()
allembs = model.tok_embeddings(allonehot.to(device)).squeeze().cpu().float()
predicted_allembs = torch.matmul(allembs, inverse).topk(k=1, dim=-1)[1].squeeze()
print("reversal acc for allembs(%)", 100 * (sum(predicted_allembs==allonehot.cpu()) / len(allonehot)).item())
print("samples", predicted_allembs.cpu().numpy()[:10], allonehot.cpu().numpy()[:10])