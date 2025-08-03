import argparse
import torch
import csv
import random
import os
import numpy as np
import logging
from Joint import HADMIL as MIL   

def parse_args_and_save():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--label_frac", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="Camelyon16")
    parser.add_argument("--pretrain", type=str, default="ResNet50_ImageNet")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--LIPN_lr", type=float)
    parser.add_argument("--DMIN_lr", type=float)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--init_type", type=str, default='default')
    parser.add_argument("--model", type=str, default='v1')
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--pretrain_dir", type=str, default='')
    parser.add_argument("--lwc", type=str, default='')
    parser.add_argument("--distill_loss", type=str, default='')
    parser.add_argument("--LIPN_ema", type=float, default=0.0)
    parser.add_argument("--num_moe", type=int, default=2)
    parser.add_argument("--attn_ratio", type=float, default=0.)
    parser.add_argument("--gpu_id", type=int, default=1)
    args = parser.parse_args()
    args = init_args(args)


    return args

    
def init_args(args):
    args.device = torch.device('cuda')
    assert args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC']
    args.feature_dir = f'/home/ubuntu/dongjy/02.data/02.processed_data/{args.dataset}/20x/feats/'
    args.label_csv = f'/home/ubuntu/dongjy/02.data/02.processed_data/{args.dataset}_label.csv'
    args.split_dir = f'/home/ubuntu/dongjy/02.data/02.processed_data/{args.dataset}/splits/'
    args.n_classes = 3 if args.dataset == 'TCGA-RCC' else 2
    args.subtyping = False if args.dataset == 'Camelyon16' else True
    args.k_sample = 8 if args.dataset == 'TCGA-RCC' else 32
    return args

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_loggers(stdout_txt):
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(stdout_txt)
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel('INFO')

def make_expdir_and_logs(args):
    dataset = {
        'Camelyon16': 'C',
        'TCGA-NSCLC': 'N',
        'TCGA-BRCA': 'B',
        'TCGA-RCC': 'R'
    }
    pretrain = {
        'ResNet50_ImageNet': 'Res50'
    }

    
    extension = f'_attn_ratio={args.attn_ratio}_num_moe={args.num_moe}_LIPN_ema={args.LIPN_ema}'



    exp_dir = os.path.join('experiments/{}{}/{}/init_{}/wrong_label_rate=0.0/label_frac={}/model={}/degree={}_distlosss={}/LIPN_lr={}_DMIN_lr={}_maskratio={}_{}{}/'.
        format(dataset[args.dataset], args.fold, pretrain[args.pretrain], \
        args.init_type, \
        args.label_frac, args.model, args.degree, args.distill_loss, args.LIPN_lr, args.DMIN_lr, args.mask_ratio, args.lwc, extension)
    )


    args.exp_dir = exp_dir
    args.log_dir = os.path.join(args.exp_dir, 'logs')

    if os.path.exists(args.exp_dir):
        if args.k == 0 and args.phase == 'train':
            pass
    else:    
        os.makedirs(args.exp_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_loggers(os.path.join(args.log_dir, "{}-stdout-fold{}.txt".format(args.phase, args.k)))


    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp dir = {}".format(args.exp_dir))
    logging.info("Writing log file to {}".format(os.path.join(args.exp_dir, 'logs')))

def main(args):
    if args.phase == 'train':
        csv_name = 'valid_metrics.csv'
    elif args.phase == 'test':
        csv_name = 'test_metrics.csv'
    else:
        raise NotImplementedError
        
    with open(os.path.join(args.log_dir, csv_name), 'a') as f:
        writer = csv.writer(f)

        if args.k == 0:
            if args.phase == 'train':
                writer.writerow(['fold', 'valid_loss', 'valid_auc', 'valid_acc', 'valid_precision', 'valid_recall', 'valid_f1', 'score_rate'])
            else:
                writer.writerow(['fold', 'test_loss', 'test_auc', 'test_acc', 'test_precision', 'test_recall', 'test_f1', 'score_rate'])

        logging.info("\n{}start fold {} {}".format(''.join(['*'] * 50), args.k, ''.join(['*'] * 50)))

        args.ckpt_dir = os.path.join(args.exp_dir, 'ckpts/fold-{}'.format(args.k))
        if not os.path.exists(args.ckpt_dir):   
            os.makedirs(args.ckpt_dir)
       
             

        MIL_runner = MIL(args)

        if args.phase == 'train':
            loss, auc, acc, precision, recall, f1, score_rate = MIL_runner.train()
        elif args.phase == 'test':
            loss, auc, acc, precision, recall, f1, score_rate = MIL_runner.test()
        else:
            raise NotImplementedError

        writer.writerow(['{}'.format(args.k)] + [round(loss, 4), round(auc, 4), round(acc, 4), round(precision, 4), round(recall, 4), round(f1, 4), round(score_rate, 4)])

if __name__ == '__main__':
    args = parse_args_and_save()
    seed_torch(0)
    make_expdir_and_logs(args)
    main(args)
