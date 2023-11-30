# Project Notes

## Work Plan


- [X] see what I commited to in the proposal
    * data pipeline
    * models to train
    * training pipeline
- [X] see the doc for requirements for check in
    * 3-5 sentences on progress
    * 3-5 sentences on roadblocks
    * UN-cleaned code 


- [ ] see requirements for the final submission too to understand HOW the deliverables right now should look
 - [ ] how should data look?
 - [ ] how commented / understandable / simple (to 6120 standard) the code should be?
- [ ] how can I __minimize__ the scope of the project given how long it takes to train and optimize the hyperparam space?
  - maybe only K-1 token?
  - maybe only FIRST K (just _the_ first?) layers?

To make:
- run ONE linear model run with my winning params
- ❗️bug fix:
  - reduced number of workers from 12 to 0 ❗️ seems to be the crux! 
      - [ ] need to figure out how the number of workers affects the parallelization
        - ??? what workers mean?
        - ??? how do they work together with runtorch?
        - ??? how many of them can i safely run? seems like no more than 2 (or 3?)
  - removed a loop from sh script
  - replaced runtorch w python in sh script


- ❓❓❓ why is `batch_idx` in train_epoch not defined (a null)? 
  - it _is_ unrelated to the num of workers (even with 1 worker)
  - ❗️seems like the solution was in making the batch size lower (below 10, I guess)
    - ??? why ?

- IMPLEMENT ALL MODELS!!!!!!
- clean up the training file
  - get rid of shuffle?
  - ??? do I need to change warmup_scheduler = warmup.UntunedLinearWarmup(optimizer) ???

- ??? what is a warmup scheduler and why i need it?

- maybe go with below instead???

- ??? got this error at the end of the run - what is it about?
```
Traceback (most recent call last):                                                                            
  File "/share/u/troitskiid/.conda/envs/troitskiid/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3652, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 147, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 176, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7080, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'decoded_prefix'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/share/u/troitskiid/projects/lexicon_light/scripts/train_probe.py", line 318, in <module>
    main(args)
  File "/share/u/troitskiid/projects/lexicon_light/scripts/train_probe.py", line 280, in main
    test_loss, test_acc, test_topk_acc, test_results = test(probe, test_loader, criterion, return_results=True)
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/share/u/troitskiid/projects/lexicon_light/scripts/train_probe.py", line 141, in test
    for (data, targets, currs, doc_idxs) in baukit.pbar(test_loader):
  File "/share/u/troitskiid/.conda/envs/troitskiid/lib/python3.11/site-packages/tqdm/std.py", line 1178, in __iter__
    for obj in iterable:
  File "/share/u/troitskiid/.conda/envs/troitskiid/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 633, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/share/u/troitskiid/.conda/envs/troitskiid/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 677, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/share/u/troitskiid/.conda/envs/troitskiid/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/share/u/troitskiid/.conda/envs/troitskiid/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 51, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/share/u/troitskiid/projects/lexicon_light/modules/state_data.py", line 188, in __getitem__
    inp, trout, layer_names = self.get_model_trace(str(doc['decoded_prefix']))
                                                       ~~~^^^^^^^^^^^^^^^^^^
  File "/share/u/troitskiid/.conda/envs/troitskiid/lib/python3.11/site-packages/pandas/core/series.py", line 1007, in __getitem__
    return self._get_value(key)
           ^^^^^^^^^^^^^^^^^^^^
  File "/share/u/troitskiid/.conda/envs/troitskiid/lib/python3.11/site-packages/pandas/core/series.py", line 1116, in _get_value
    loc = self.index.get_loc(label)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/share/u/troitskiid/.conda/envs/troitskiid/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3654, in get_loc
    raise KeyError(key) from err
KeyError: 'decoded_prefix'
```

- [ ] understand the data pipeline
  - ??? where are the activations stored?
  - ??? how do we know which token is the prefix to the current one?
  - ??? where are we sourcing the information from?
- [ ] comment with chatgpt state_data.py
- [ ] Rewrite the state_data.py (change data file such that it ONLY contains the prefix text and nothing else (or maybe only reference to pile doc?)
- [ ] change module structure? move state_data into modules.py?
- [ ] Rewrite the train script?
      * maybe not a fully working now? no need to test...
    - [ ] ❗️ TRY to run
    - [ ] simplify (get rid of warmups and other fancy training stuff (check if they had anything like that in the class))

- [ ] comment! - @based on Sheridan's code
- [ ] ❗️ remove dev COMMENTS

- [ ] reimplement random sampling of the pile from the stored pile dataset just like Sheridan did (that is the way to create a dataset - and I can do it in a jupyter notebook)
    - leave the cells state as ran
    - change paths to koyenas files to stub paths to pile dataset (explain how it should look / be structured)

- [ ] add Xavier initialization to rnn and mlp? (see chatgpt). I have Xavier as random option for the linear model

Report:
- created data (randomly sample 500 for val, test and train from downloaded pile)
- created linear model 
- created the training pipeline
  - set up wandb 

Roadblocks:
- bad training; doing the sanity check + pinv init
- not enough time to implement beyond linear?
- pipeline but not training?
- ??? report on the cuda memory issued?


Further goals
- [ ] see what we have used in the course for model / ngram training to know the limits ???
- [ ] change project structure to something simpler the way we have in 6120 project?
- [ ] create a pynotebook with experiments!
  - ??? model it upon Sheridan's scripts/explore.ipynb ?
- [ ] Write a report


REMOVED:
- weighted_mse from train_gptj_probe and training.py
- init_matrix from training->LinearModel and train_gptj_probe

## Analysis of current project (nocache branch)

modules/
- ??? helpers.py - what do i need this one for?
- ??? state_data.py - is this where the datasets are made?
  - [ ] find where it's used
- training.py
  - [ ] simplify the LinearModel class
    - [ ] remove the init_matrix param?
      - ??? though maybe there is a chance I'd use it if I want to instantiate from the pinv / trained embed-to-token matrix?
scripts/
- ??? ngrams.py

Probs can remove:
- modules/entity_data.py - seems to just be related to counterfact
- scripts/counterfact.py
- scripts/explore.ipynb
- scripts/initial_sweep.sh
- scripts/ngrams.py
- scripts/noshufhailmary.sh
- scripts/shufhailmary.sh
- scripts/pinv_baseline.py
- scripts/replicate.sh
- scripts/run_probes.sh
- scripts/sanity_probe.sh
- scripts/sanity_probe.py
- scripts/test.sh