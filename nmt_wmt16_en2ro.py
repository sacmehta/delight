import os
import argparse
from fairseq.delight_modules.print_utilities import *

# LR FOR d_m=512. For other dims, we will scale it linearly
LR_512 = 0.0007
TESTED_DIMS = [128, 256, 384, 640]


def run_experiment(args):
    max_update = args.max_updates
    warmup_update = args.warmup_updates
    max_tokens = args.max_tokens
    update_freq = args.update_freq
    num_gpus = args.num_gpus
    data_dir = args.data_dir
    results_dir = args.save_dir

    # scale LR
    d_m = args.d_m

    if d_m not in TESTED_DIMS:
        print_warning_message('We have only tested for {}. Got {}'.format(TESTED_DIMS, d_m))

    max_lr = min(round((512.0 /d_m) * LR_512, 4), 0.01)

    job_name = 'delight_out_{}'.format(d_m)

    results_dir = '{}/{}'.format(results_dir, job_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    log_file = '{}/logs.txt'.format(results_dir)

    command = ['python train.py {} --arch delight_transformer_wmt16_en_ro '
               '--no-progress-bar '
               '--optimizer adam --adam-betas \'(0.9, 0.98)\' --clip-norm 0.0 --weight-decay 0.0 '
               '--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --min-lr 1e-09 '
               '--update-freq {} --keep-last-epochs 10 '
               '--ddp-backend=no_c10d --max-tokens {} '
               '--max-update {} --warmup-updates {} '
               '--lr-scheduler linear --warmup-init-lr 1e-7 --lr {} --min-lr 1e-9 '
               '--t-mult 1 --save-dir {} '
               '--distributed-world-size {} --distributed-port 50786 '
               '--delight-emb-map-dim 128 --delight-emb-out-dim {} '
               '--delight-enc-min-depth 4 --delight-enc-max-depth 8 --delight-enc-width-mult 2 '
               '--delight-dec-min-depth 4 --delight-dec-max-depth 8 --delight-dec-width-mult 2 '
               '| tee -a {}'.format(data_dir,
                                    update_freq, max_tokens,
                                    max_update, warmup_update, max_lr,
                                    results_dir,
                                    num_gpus,
                                    d_m,
                                    log_file
                                    )]

    print_log_message('Training command: ')
    print(command[0])
    os.system(command[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training script for WMT14 En2De')

    parser.add_argument('--d-m', type=int, default=128, help='Model dimension')
    parser.add_argument('--data-dir', type=str, default='data-bin/wmt14_en_ro', help='Data location')
    parser.add_argument('--max-updates', type=int, default=100000, help='Max. updates')
    parser.add_argument('--warmup-updates', type=int, default=10000, help='Warmup updates')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Max. tokens')
    parser.add_argument('--update-freq', type=int, default=1, help='update freq')
    parser.add_argument('--num-gpus', type=int, default=8, help='num. of GPUs')
    parser.add_argument('--save-dir', type=str, default='./results_wmt16_en2ro', help='Results directory')


    args = parser.parse_args()

    run_experiment(args)
