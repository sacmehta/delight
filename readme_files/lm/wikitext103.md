
# Pre-processing, Training, and Evaluation on WikiText-103 dataset

This file describes the steps for (1) downloading dataset, (2) processing dataset, (3) training, and (4) evaluation.

## Dataset download and pre-processing

From the main directory, run the following command to download the dataset:

``` 
    cd examples/language_model/
    bash prepare-wikitext-103.sh
    cd ../..
```

To pre-process the dataset, run the following command:

``` 
TEXT=examples/language_model/wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

## Training

To train a model with a single node comprising of 8 V100 GPUs (each with 32 GB memory), you can use the following command:

``` 
python lm_wikitext_103.py --d-m 256
```

where `--d-m` is the model dimension. In our experiments, we have only tested `d-m={128, 256, 384, 512, 1024}`
 


## Evaluation

To evaluate a model, you can use the following command:

```
python eval_lm.py data-bin/wikitext-103 --path <checkpoint_dir>/checkpoint_best.pt --max-sentences 2 --tokens-per-sample 512 --context-window 400 --gen-subset test --res-file eval_logs.txt
```