'''Utility functions helpful for model training.
'''
# partially adapted from https://github.com/sfeucht/lexicon and modified as part of a research project at David Bau's lab
import torch

def _topktoks(hs, k=1):
    """
    Returns the top k tokens from the given hidden state.

    Parameters:
    hs (torch.Tensor): Input hidden state tensor of shape (batch_size, sequence_length, num_classes).
    k (int): The number of top tokens to return. Default is 1.

    Returns:
    torch.Tensor: The top k tokens.
    """
    _, top_tokens = hs.topk(k=k, dim=-1)
    return top_tokens


def _topkprobs(hs, tokenizer, k=5):
    """
    Compute the top-k probabilities and corresponding tokens from the given hidden state.

    Args:
        hs (torch.Tensor): Input hidden state tensor of shape (batch_size, sequence_length, num_classes).
        tokenizer: Tokenizer object used to decode the tokens.
        k (int): Number of top probabilities to compute (default: 5).

    Returns:
        dict: A dictionary containing the top-k probabilities and tokens.
            The keys are in the format "top_i_prob", "top_i_tok_id", and "top_i_tok",
            where i ranges from 1 to k.
    """
    top_probs, top_tokens = torch.softmax(hs, dim=0).topk(k=k, dim=-1)
    out = {}
    for i in range(k):
        out[f"top_{i+1}_prob"] = top_probs[i].item()
        out[f"top_{i+1}_tok_id"] = top_tokens[i].item()
        out[f"top_{i+1}_tok"] = tokenizer.decode(top_tokens[i].tolist())
    return out