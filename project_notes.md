# Code 
  

## Make it my own
- [X] modify sanity_check
  - [X] rename files here (baseline instead of sanity check)

- [X] ChatGPT - COMMENT THROUGH all scripts
- [X] Understand and edit comments  
  - [X] How data works
  - [X] warm up logic
- [X] add instructions to the top ("Can we run your code by following the instructions at the top of your jupyter notebook/main python file?")
  - [X] test and mention how much memory we need to run on batch size 1
- [X] mention in .sh script how long training for each layer on _which_ gpu takes
- [X] move baseline dataset and collate fns to
- [X] Simplify 
  - [X] remove _complex list comprehension_ etc
  - [X] change module structure? move state_data into modules.py?
- [X] Clean up
  - [X] ❗️ remove dev COMMENTS
  - [X] clean FIXMEs
  - [X] do not commit wandb logs and checkpoints
- [ ] ❗️ TEST both scripts before submitting

- ??? create visualizations of wandb evaulations via seaborn??
- [ ] create jupyter notebook that samples the data OR MAYBE NOT (nnot super important)
  - [ ] reimplement random sampling of the pile from the stored pile dataset just like Sheridan did (that is the way to create a dataset - and I can do it in a jupyter notebook)
    - leave the cells state as ran
    - change paths to koyenas files to stub paths to pile dataset (explain how it should look / be structured)

- [ ] save sweep set ups as .yaml files?


# Report
See and add tasks in [GOOGLE DOC](https://docs.google.com/document/d/1LhdHEw8qLpkogLPIaymseeDFDyasFhjjpoe7gth0tm8/edit)


# Slides and Video
See and add tasks in [GOOGLE DOC](https://docs.google.com/document/d/1HG5AE1hmcunyZuqOT6LRYiIyh_MSzpA-xpxxt5fgX3E/edit)

==============================================================================================

# More stuff

## Evaluate / Data Analysis
- [X] EVALUATE p/r/f1 (or ROUGE, BLEU, or a different metric appropriate for your task)
- [X] make wandb graph of layer/test_acc corelation
- "As a result to present for this project I will be producing loss graphs over at least 5 training epochs for each model. I will also produce a joint graph comparing the performance of the three models."

- [ ] Do a data analysis / compariosn of results between mmodels / layers in a jupyter notebook?

## Run Linear on all layers
- [ ] ❗️ gotta run on layer -1 for test !
- [X] how many layers are there in gpt-j?
  - 28 it seems! (hugging face model card)
- [X] use .sh script in tmux %%ended up using a wandb sweep%%
- [X] use best hyperparams so far
  - using this run https://wandb.ai/baulab/lexicon-cat-probes/runs/bkcrffb2/overview?workspace=user-tentative

## Establish Baseline FFN and RNN
- mlp sanity probe takes 16706MiB to run
- rnn sanity probe takes 18358MiB to run

- [X] PAUSE linear over all layers
- [X] make an interesting version of rnn
- [X] run rnn baseline sweep

- [ ] gather the successful linear probe sanity check run sweep data for chat gpt prompt
- [ ] ask chatgtpt good params for mlp and rnn
  - [X] add Xavier initialization to rnn and mlp? (see chatgpt). I have Xavier as random option for the linear model
  - [X] prompt: "with advanced understanding of deep learning theory and mathematics (give example of lectures)"
  - [X]  do I need to change warmup_scheduler = warmup.UntunedLinearWarmup(optimizer) ??? %%yeah, it seems fine%%%
  - [ ] if very detailed prompt with all the code doesn't help, present a description of training set up, the sweep results and describe model configs (vocab size values etc)
  - describe set up
  - describe current linear params
  - describe hyperparams
  - describe results
  - 🔥 look at what it suggested before for each of mlp and rnn
    -  [ ] rename the chatgpt chats for easier search
- ??? which parameters to Sweep Over ?
- [X] implement base MLP and RNN
  - [ ] see recommendation on sizes from coursework
  - [X] use CURRENT PARAMS ::as default::

- [X] ❗️❗️❗️ establish BASELINE for MLP and RNN 
  - [X] set up wider (than optimal linear for sanity) but stricter sweep ranges
    - try SGD again?
  
- [ ] ❓should I enable shuffle with the linear probe to increase accuracy on test data? (it currently might be overfitting on train)

## Fully train 
- [X] train mlp on layer 0 with basic successful hypers
  - [ ] do a sweep over additional chatgpt suggested features - see results

- [X] take baseline params and train FFN and RNN on 0 layer
- [X] use best on 0 and train on 1-32 (or how many?)

==============================================================================================

- ❗️bug fix:
  - reduced number of workers from 12 to 0 ❗️ seems to be the crux! 
      - [ ] need to figure out how the number of workers affects the parallelization
        - ??? what workers mean?
        - ??? how do they work together with runtorch?
        - ??? how many of them can i safely run? seems like no more than 2 (or 3?)
  - removed a loop from sh script
  - replaced runtorch w python in sh script

REMOVED:
- weighted_mse from train_gptj_probe and training.py
- init_matrix from training->LinearModel and train_gptj_probe


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