import argparse
import pandas as pd
from itertools import islice
from collections import Counter
from llama import Tokenizer
import matplotlib.pyplot as plt


tokenizer = Tokenizer(model_path="../llama/tokenizer.model")
min_appearances = 10

def tuple_to_s(bg):
    return repr(f"{tokenizer.decode(bg[0])} {tokenizer.decode(bg[1])}")

def see(bigram, extra=''):
    print(repr(tokenizer.decode(bigram[0])), repr(tokenizer.decode(bigram[1])), extra)

def interpret_counter(ctr, top=100, title=''):
    print(f"{len(ctr)} unique {title} bigrams that appear more than {min_appearances} times")
    print(f"top {top} bigrams:")
    for (a, b), ct in ctr.most_common(top):
        print(repr(tokenizer.decode(a)), repr(tokenizer.decode(b)), f"count: {ct}")

def main(args):
    # first, count up all of the bigrams in the TESTING set.  
    df = pd.read_csv(args.gt)
    gt_ctr = Counter()
    all_gt_ct = 0
    for d in list(df['decoded_prefix']):
        toks = tokenizer.encode(d, bos=True, eos=False)
        gt_ctr += Counter(zip(toks, toks[1:]))
        all_gt_ct += len(toks)
    
    gt_ctr = Counter({k:v for k, v in gt_ctr.items() if v >= min_appearances})
    # interpret_counter(gt_ctr, top=20, title=f"gt_ctr({all_gt_ct})")

    # count up all the bigrams of a given PROBE 
    preds = pd.read_csv(args.results).fillna('')
    corr = preds.loc[preds['predicted_tok_id'] == preds['actual_tok_id']]
    probe_ctr = Counter(zip(corr['actual_tok_id'], corr['current_tok_id']))
    probe_ctr = Counter({k:v for k, v in probe_ctr.items() if v >= min_appearances})
    interpret_counter(probe_ctr, top=20, title=f"probe_ctr({len(preds)})")

    # see which bigrams the probe ALWAYS got wrong. 
    most_common_gt = dict(gt_ctr.most_common())
    most_common_pr = dict(probe_ctr.most_common())
    probe_missed_all = []
    for bigram, ct in most_common_gt.items():
        if bigram not in most_common_pr.keys():
            if ct > 1:
                probe_missed_all.append((bigram, ct))
    print(f"the probe always missed {len(probe_missed_all)} unique bigrams appearing more than once")
    for m, ct in probe_missed_all:
        see(m, f"there were {ct}")
        

    # see which bigrams the probe ALWAYS got right.  
    probe_recovered_all = {}
    for bigram, ct in most_common_pr.items():
        if most_common_gt[bigram] == ct:
            probe_recovered_all[bigram] = ct
    # print(f"the probe always recovered {len(probe_recovered_all)} unique bigrams appearing more than once")
    # for bg,ct in probe_recovered_all.items():
    #     if ct > 1: see(bg, extra=ct) 
        
    # calculate aggregate percentage of gt_bigrams>1 that were recovered by probe
    # print(f"that's {len(probe_recovered_all) / len([v for v in most_common_gt.values() if v > 1])}% of all bigrams that appeared more than once in test set")

    # create a stacked bar chart in order of appearance, where blue (larger) is the
    # ground truth number of bigram in the testing set and red (smaller) is the number of times
    # that linear probe recovered that bigram. 
    xticklabels = [tuple_to_s(bg) for bg, ct in gt_ctr.most_common(100)]
    gt_values = [ct for _, ct in gt_ctr.most_common(100)]
    probe_values = []
    for bg, ct in gt_ctr.most_common(100):
        try:
            probe_values.append(most_common_pr[bg])
        except KeyError:
            probe_values.append(0)
    
    # Create a stacked bar graph
    fig, ax = plt.subplots()

    labels = [label if p / gt < 0.6 else '' for label, gt, p in zip(xticklabels, gt_values, probe_values)]
    ax.bar(xticklabels, gt_values, label='Ground Truth Count', tick_label = labels)
    ax.bar(xticklabels, probe_values, label='Probe Count')

    # xticklabels = [label if p / gt < 0.6 else '' for label, gt, p in zip(xticklabels, gt_values, probe_values)]
    # ax.set_xticks(gt_values)
    # ax.set_xticklabels(xticklabels)
    plt.xticks(rotation=90, fontsize=8)
    print([label for label in labels if label != ''])

    # Add labels and legend
    ax.set_xlabel('Bigrams')
    ax.set_ylabel('Count')
    ax.set_title('Proportion of ground truth bigrams recovered by linear probe')
    ax.legend()
    plt.tight_layout()
    plt.savefig("bgs_recovered.png", dpi=100, format="png")

    



    # LATER TODO 
    # 3. "are certain bigrams always right and some usually wrong?" for all the gt bigrams
    #    that are >threshold occurrences, plot the distribution of % correct for probe preds.
    # 4. animation across layers
    # 5. qualitative analysis across layers
    # 6. analysis on the training ngrams as well. want to see if maybe that explains why the probe picks up on outliers in test dataset 





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default="../data/test_tiny_500.csv")
    parser.add_argument('--results', type=str, default="../logs/llama-2-7b/LAYER29-TGTIDX-1-train_small_5000-bsz10-lr0.00473-epochs60-9vqlvyf0/test_tiny_500_results.csv") # good last layer one
    # parser.add_argument('--results', type=str, default="../logs/llama-2-7b/LAYER3-TGTIDX-1-train_small_5000-bsz1-lr0.00608-epochs74-ak6qyq6v/test_tiny_500_results.csv") # pretty good early layer one
    # parser.add_argument('--results', type=str, default="../logs/llama-2-7b/LAYER5-TGTIDX-1-train_small_5000-bsz5-lr0.00370-epochs25-3o0cvtkg/test_tiny_500_results.csv") # bad probe
    
    args = parser.parse_args()
    main(args)