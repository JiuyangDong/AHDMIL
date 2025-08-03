import os
import argparse
import time 

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int)
parser.add_argument("--LIPN_lr", type=float)
parser.add_argument("--DMIN_lr", type=float)
parser.add_argument("--fold", type=int, default=10)
parser.add_argument("--label_frac", type=float, default=1.00)
parser.add_argument("--pretrain", type=str, default='Res50_on_ImageNet')
parser.add_argument("--dataset", type=str, default='Camelyon16')
parser.add_argument("--init_type", type=str, default='normal')
parser.add_argument("--mask_ratio", type=float, default=0.1)
parser.add_argument("--degree", type=int, default=0)
parser.add_argument("--model", type=str, default='v1')
parser.add_argument("--pretrain_dir", type=str, default='Null')
parser.add_argument("--lwc", type=str, default='Null')
parser.add_argument("--distill_loss", type=str, default='Null')
parser.add_argument("--pse_weight", type=float, default=1.)
parser.add_argument("--LIPN_ema", type=float, default=0.0)
parser.add_argument("--num_moe", type=int, default=2)
parser.add_argument("--attn_ratio", type=float, default=0.)
args = parser.parse_args()

#######################################################
# train and val
#######################################################
for k in range(10):
    cmd = 'CUDA_VISIBLE_DEVICES={} python baseline.py --phase train --dataset {} --pretrain {} --model {} --mask_ratio {} \
     --fold {} --label_frac {} --LIPN_lr {} --DMIN_lr {} --k {} --degree {} --init_type {} --pretrain_dir {} --lwc {} --distill_loss {} \
     --LIPN_ema {} --num_moe {} --attn_ratio {} --gpu_id {}'.format(
        args.gpu_id, args.dataset, args.pretrain, args.model, args.mask_ratio, args.fold, args.label_frac, args.LIPN_lr, args.DMIN_lr, k, args.degree, args.init_type, args.pretrain_dir, args.lwc, args.distill_loss, \
        args.LIPN_ema, args.num_moe, args.attn_ratio, args.gpu_id)
    os.system(cmd)

    cmd = 'CUDA_VISIBLE_DEVICES={} python baseline.py --phase test --dataset {} --pretrain {} --model {} --mask_ratio {} \
     --fold {} --label_frac {} --LIPN_lr {} --DMIN_lr {} --k {} --degree {} --init_type {} --pretrain_dir {} --lwc {} --distill_loss {} \
     --LIPN_ema {} --num_moe {} --attn_ratio {} --gpu_id {}'.format(
        args.gpu_id, args.dataset, args.pretrain, args.model, args.mask_ratio, args.fold, args.label_frac, args.LIPN_lr, args.DMIN_lr, k, args.degree, args.init_type, args.pretrain_dir, args.lwc, args.distill_loss, \
        args.LIPN_ema, args.num_moe, args.attn_ratio, args.gpu_id)
    os.system(cmd)
