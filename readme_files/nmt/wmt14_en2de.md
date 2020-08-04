
# Pre-processing, Training, and Evaluation on WMT'14 En-De dataset

This file describes the steps for (1) downloading dataset, (2) processing dataset, (3) training, and (4) evaluation.

## Dataset download and pre-processing

From the main directory, run the following command:

``` 
bash prepare_nmt_dataset.sh wmt14_en_de
```

## Training

To train a model with a single node comprising of 8 V100 GPUs (each with 32 GB memory), you can use the following command:

``` 
python nmt_wmt14_en2de.py --d-m 256
```

where `--d-m` is the model dimension. In our experiments, we have only tested `d-m={128, 256, 384, 512, 640}`
 


## Evaluation

To evaluate a model, you can use the following command:

```
GEN_RES_FILE=gen_out.out
python generate.py data-bin/wmt14_en_de/ --path <results_dir>/checkpoint_best.pt --beam 5 --lenpen 0.4 --remove-bpe --batch-size 128 > GEN_RES_FILE

bash scripts/compound_split_bleu.sh GEN_RES_FILE 
```

## Results
Here are the results that we obtain.

| Model dimension (d_m) | Parameters | BLEU | Training Logs |
| --------------------- | ---------- | ---- | ------------- |
| 128 | 8.09 M | 22.7 | [Link](https://gist.github.com/sacmehta/f1464bac0491efad36cb1e4b620bc4b7#file-delight_wmt14_en2de_dm_128-txt) |
| 256 | 13.79 M | 25.5 | [Link](https://gist.github.com/sacmehta/f1464bac0491efad36cb1e4b620bc4b7#file-delight_wmt14_en2de_dm_256-txt) |
| 384 | 23.25 M | 26.7 | [Link](https://gist.github.com/sacmehta/f1464bac0491efad36cb1e4b620bc4b7#file-delight_wmt14_en2de_dm_384-txt) |
| 512 | 36.76 M | 27.6 | [Link](https://gist.github.com/sacmehta/f1464bac0491efad36cb1e4b620bc4b7#file-delight_wmt14_en2de_dm_512-txt) |
| 640 | 54.04 M | 28.0 | [Link](https://gist.github.com/sacmehta/f1464bac0491efad36cb1e4b620bc4b7#file-delight_wmt14_en2de_dm_640-txt) |