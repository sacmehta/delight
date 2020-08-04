
# Pre-processing, Training, and Evaluation on WMT'16 En-Ro dataset

This file describes the steps for (1) downloading dataset, (2) processing dataset, (3) training, and (4) evaluation.

## Dataset download and pre-processing

Download the pre-processed dataset by clicking this [link](https://drive.google.com/file/d/1kISrB2ecKzQDuS0N67iRkTjuTs19oZ5j/view?usp=sharing).


## Training

To train a model with a single node comprising of 8 V100 GPUs (each with 32 GB memory), you can use the following command:

``` 
python nmt_wmt16_en2ro.py --d-m 384
```

where `--d-m` is the model dimension. In our experiments, we have only tested `d-m={128, 256, 384, 640}` 


## Evaluation

To evaluate a model, you can use the following command:

```
python generate.py data-bin/wmt16_en_ro/ --path <results_dir>/checkpoint_best.pt --beam 5 --remove-bpe --batch-size 128 --quiet
```

## Results
Here are the results that we obtain.

| Model dimension (d_m) | Parameters | BLEU | Training Logs |
| --------------------- | ---------- | ---- | ------------- |
| 128 | 6.97 M | 32.0 | [Link](https://gist.github.com/sacmehta/57c12358434f12bf15939311469c7173#file-delight_wmt16_en2ro_dm_128-txt) |
| 256 | 12.67 M | 33.8 | [Link](https://gist.github.com/sacmehta/57c12358434f12bf15939311469c7173#file-delight_wmt16_en2ro_dm_256-txt) |
| 384 | 22.12 M | 34.3 | [Link](https://gist.github.com/sacmehta/57c12358434f12bf15939311469c7173#file-delight_wmt16_en2ro_dm_384-txt) |