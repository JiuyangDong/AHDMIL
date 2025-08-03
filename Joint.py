import torch
import pandas as pd
import os
import itertools
import logging
import json
import itertools
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler, Dataset
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F 
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append('..')
from tqdm import tqdm
import random 
from copy import deepcopy
import math
import time 
import subprocess

class DualWSIDataset(Dataset):
    def __init__(self, args, wsi_labels, infold_cases, phase=None):
        self.args = args
        self.wsi_labels = wsi_labels
        self.phase = phase
        

        self.low_level_dir = f'/home/ubuntu/dongjy/02.data/02.processed_data/{args.dataset}/1.25x_img_pt/'

        self.infold_features, self.infold_labels = [], []
        for case_id, slide_id, label in wsi_labels:
            if case_id in infold_cases:
                if args.pretrain in ['ResNet50_ImageNet']:
                    if self.args.dataset in ['Camelyon16', 'Camelyon17', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC']:
                        fea_path = os.path.join(args.feature_dir, args.pretrain, slide_id+'.pt')
                        if os.path.exists(fea_path):
                            
                            self.infold_labels.append(label)

                            img_path = self.low_level_dir + slide_id+'.pt'
                            assert os.path.exists(img_path)

                            

                            self.infold_features.append([fea_path, img_path])
                        
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                

    def __len__(self):
        return len(self.infold_features)
        
    def __getitem__(self, index):
        fea_path, img_path = self.infold_features[index]
        label = self.infold_labels[index]
        fea = torch.load(fea_path)
        img = torch.load(img_path)[:,:,:,:3]
        assert fea.shape[0] == img.shape[0]
        return fea, label, fea_path, img, img_path

class HADMIL:
    def __init__(self, args):
        self.args = args

        if args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC']:
            self.train_loader, self.valid_loader, self.test_loader = self.init_data_wsi()
        else:
            raise NotImplementedError
        
        self.multi_init()
        
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.cross_entropy_loss_ls = torch.nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

        self.counter = 0
        self.patience = 5
        self.stop_epoch = 0
        self.best_loss = np.Inf
        self.flag = 1

        self.LIPN_path = os.path.join(self.args.ckpt_dir, 'best_LIPN.pth')
        self.DMIN_path = os.path.join(self.args.ckpt_dir, 'best_DMIN.pth')
        self.best_valid_metrics = None 

        self.lower_ratio = 0.3
        self.upper_ratio = 1.0

    def multi_init(self):
        self.DMIN, self.LIPN = self.init_model()
        self.pretrained_pretrain_path = os.path.join(self.args.pretrain_dir, 'fold-{}/best_epoch.pth'.format(self.args.k))
        print(self.DMIN.load_state_dict(torch.load(self.pretrained_pretrain_path)))
        
        self.optimizer1 = torch.optim.Adam(self.LIPN.parameters(), lr=self.args.LIPN_lr, weight_decay=self.args.wd)
        self.optimizer2 = torch.optim.Adam(self.DMIN.parameters(), lr=self.args.DMIN_lr, weight_decay=self.args.wd)

    def read_wsi_label(self):
        data = pd.read_csv(self.args.label_csv)

        wsi_labels = []
        for i in range(len(data)):
            case_id, slide_id, label = data.loc[i, "case_id"], data.loc[i, "slide_id"], data.loc[i, "label"]

            if self.args.dataset in ['Camelyon16']:
                assert label in ['tumor_tissue', 'normal_tissue']
                label = 0 if label == 'normal_tissue' else 1
            elif self.args.dataset == 'TCGA-NSCLC':
                assert label in ['TCGA-LUSC', 'TCGA-LUAD']
                label = 0 if label == 'TCGA-LUSC' else 1
            elif self.args.dataset == 'TCGA-BRCA':
                assert label in ['Infiltrating Ductal Carcinoma', 'Infiltrating Lobular Carcinoma']
                label = 0 if label == 'Infiltrating Ductal Carcinoma' else 1
            elif self.args.dataset == 'TCGA-RCC':
                assert label in ['TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP']
                if label == 'TCGA-KICH':
                    label = 0
                elif label == 'TCGA-KIRC':
                    label = 1
                elif label == 'TCGA-KIRP':
                    label = 2
            else:
                raise NotImplementedError

            wsi_labels.append([case_id, slide_id, label])   

        return wsi_labels

    def read_in_fold_cases(self, fold_csv):
        data = pd.read_csv(fold_csv)
        train_cases, valid_cases, test_cases = data.loc[:, 'train'].dropna(axis=0, how='any').to_list(), data.loc[:, 'val'].dropna(axis=0, how='any').to_list(), data.loc[:, 'test'].dropna(axis=0, how='any').to_list()
        return train_cases, valid_cases, test_cases

    def make_weights_for_balanced_classes_split(self, data_set):
        N = float(len(data_set))           

        classes = {}
        for label in data_set.infold_labels:
            if label not in classes:
                classes[label] = 1
            else:
                classes[label] += 1
                                                                                                    
        weight = [0] * int(N)                                           
        for idx in range(len(data_set)):   
            y = data_set.infold_labels[idx]                       
            weight[idx] = N / classes[y]    
            
        return torch.DoubleTensor(weight)

    def init_data_wsi(self):
        wsi_labels = self.read_wsi_label()

        split_dir = os.path.join(self.args.split_dir, '{}-fold-{}%-label/').format(self.args.fold, int(self.args.label_frac * 100))
        train_cases, valid_cases, test_cases = self.read_in_fold_cases(os.path.join(split_dir + 'splits_{}.csv'.format(self.args.k)))
        

        train_set = DualWSIDataset(self.args, wsi_labels, train_cases, 'train')
        valid_set = DualWSIDataset(self.args, wsi_labels, valid_cases, 'valid')
        test_set = DualWSIDataset(self.args, wsi_labels, test_cases, 'test')
        
        logging.info("Case/WSI number for trainset in fold-{} = {}/{}".format(self.args.k, len(train_cases), len(train_set)))
        logging.info("Case/WSI number for validset in fold-{} = {}/{}".format(self.args.k, len(valid_cases), len(valid_set)))
        logging.info("Case/WSI number for testset in fold-{} = {}/{}".format(self.args.k, len(test_cases), len(test_set)))

        weights = self.make_weights_for_balanced_classes_split(train_set)

        if self.args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC']:
            train_loader = DataLoader(train_set, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights), replacement=True))
            valid_loader = DataLoader(valid_set, batch_size=1, sampler = SequentialSampler(valid_set))
            test_loader = DataLoader(test_set, batch_size=1, sampler = SequentialSampler(test_set))

            return train_loader, valid_loader, test_loader
        else:
            raise NotImplementedError

    def init_model(self):
        from models.hdmil import KAN_CLAM_MB_v5, SmoothTop1SVM, ImageProcessor_MoE
        DMIN = KAN_CLAM_MB_v5(I=self.args.feature_dim, dropout = True, n_classes = self.args.n_classes, subtyping = self.args.subtyping,
            instance_loss_fn = SmoothTop1SVM(n_classes = 2).cuda(self.args.device), k_sample = self.args.k_sample, args=self.args).to(self.args.device)
        LIPN = ImageProcessor_MoE(args = self.args).to(self.args.device)
        return DMIN, LIPN

    def train(self):
        step = 0


        self.LIPN.cross_EMA()        

        for epoch in range(1, self.args.n_epochs + 1):

            avg_LIPN_loss, avg_mask_rate, avg_score_rate, avg_DMIN_loss = 0, 0, 0, 0

            self.DMIN.train()
            self.LIPN.train()


            for i, (fea, label, fea_path, img, img_path) in enumerate(tqdm(self.train_loader)):

                step += 1
                fea, label, img = fea.to(self.args.device), label.to(self.args.device), img.to(self.args.device)

                self.optimizer1.zero_grad()
                LIPN_loss, mask_rate, score_rate = self.train_LIPN(fea, label, img)
                if torch.isnan(LIPN_loss):
                    continue
                avg_LIPN_loss += LIPN_loss.item()
                avg_mask_rate += mask_rate.item()
                avg_score_rate += score_rate.item()
                LIPN_loss.backward()
                self.optimizer1.step()

                self.optimizer2.zero_grad()
                DMIN_loss = self.tune_DMIN(fea, label, img)
                avg_DMIN_loss += DMIN_loss.item()
                DMIN_loss.backward()
                self.optimizer2.step()

            avg_LIPN_loss /= (i + 1)
            avg_mask_rate /= (i + 1)
            avg_score_rate /= (i + 1)
            avg_DMIN_loss /= (i + 1)
        
            logging.info("In step {} (epoch {}), average DMIN loss = {:.4f} LIPN loss = {:.4f} mask rate = {:.4f} score rate = {:.4f}".\
                        format(step, epoch, avg_DMIN_loss, avg_LIPN_loss, avg_mask_rate, avg_score_rate))
        
            self.LIPN.cross_EMA()

            self.valid(epoch)

            if self.flag == -1:
                break

        return self.best_valid_metrics

    def valid(self, epoch):
        avg_loss, avg_score = 0, 0

       
        self.LIPN.eval()
        self.DMIN.eval()


        labels, probs = [], []

        for i, (fea, label, fea_path, img, img_path) in enumerate(tqdm(self.valid_loader)):
            fea, label, img = fea.to(self.args.device), label.to(self.args.device), img.to(self.args.device)
            with torch.no_grad():
                loss, y_prob, score_rate = self.test_inference(fea, label, img, self.DMIN, self.LIPN)

            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()
            avg_score += score_rate.item()

        avg_loss /= (i + 1)
        avg_score /= (i + 1)

        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info("loss = {:.4f}, auc = {:.4f}, acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}, score rate = {:.4f}".\
            format(avg_loss, auc, acc, precision, recall, f1, avg_score))

        if avg_loss < self.best_loss and self.lower_ratio <= avg_score <= self.upper_ratio:
            self.counter = 0
            logging.info(f'Validation loss decreased ({self.best_loss:.4f} --> {avg_loss:.4f}).  Saving model ...')

            torch.save(self.DMIN.state_dict(), self.DMIN_path)
            torch.save(self.LIPN.state_dict(), self.LIPN_path)

            self.best_loss = avg_loss
            self.best_valid_metrics = [avg_loss, auc, acc, precision, recall, f1, avg_score]
        else:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if self.best_valid_metrics is not None:
                    self.flag = -1
                else:
                    self.counter = 0
                    self.multi_init()
                    if self.args.dataset == 'TCGA-RCC' and epoch > 50:
                        self.args.lower_ratio -= 0.05 


    def test(self):
        avg_loss, avg_score = 0, 0
        
        self.LIPN.load_state_dict(torch.load(self.LIPN_path))
        self.LIPN.eval()

        self.DMIN.load_state_dict(torch.load(self.DMIN_path))
        self.DMIN.eval()


        labels, probs = [], []
 
        for i, (fea, label, fea_path, img, img_path) in enumerate(tqdm(self.test_loader)):
            fea, label, img = fea.to(self.args.device), label.to(self.args.device), img.to(self.args.device)

            with torch.no_grad():
                loss, y_prob, score_rate = self.test_inference(fea, label, img, self.DMIN, self.LIPN)

            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()
            avg_score += score_rate.item()
       
        avg_loss /= (i + 1)
        avg_score /= (i+1)

        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)

        prediction_dir = os.path.join(self.args.exp_dir, 'predictions')
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        np.save(os.path.join(prediction_dir, f'AHDMIL_probs_{self.args.k}'), probs)
        np.save(os.path.join(prediction_dir, f'AHDMIL_labels_{self.args.k}'), labels)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info("loss = {:.4f}, auc = {:.4f}, acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}, score rate = {:.4f}".\
            format(avg_loss, auc, acc, precision, recall, f1, avg_score))

        return avg_loss, auc, acc, precision, recall, f1, avg_score

    def cal_score_rate(self, img_scores):
        if self.args.n_classes == 2:
            score_add = img_scores[:, :, 0] + img_scores[:, :, 1]
        elif self.args.n_classes == 3:
            score_add = img_scores[:, :, 0] + img_scores[:, :, 1] +  + img_scores[:, :, 2]
        score_union = torch.zeros_like(score_add, memory_format=torch.legacy_contiguous_format).masked_fill(score_add != 0, 1.0)
        score_rate = (score_union - score_add.detach() + score_add).sum() / (img_scores.shape[0] * img_scores.shape[1])
        
        return score_rate

    def train_single_LIPN(self, img_scores, score_rate, mask, mask_rate):
        if self.args.distill_loss == 'l1':
            loss = self.l1_loss(img_scores, mask) + self.l2_loss(score_rate, mask_rate.detach())
        else:
            raise NotImplementedError
        
        return loss

    def train_LIPN(self, fea, label, img):    
        
        if self.args.attn_ratio == 1.0:
            train_manner = 'attn_train'
        elif self.args.attn_ratio == 0.0:
            train_manner = 'hard_train'
        else:
            if np.random.uniform() <= self.args.attn_ratio:
                train_manner = 'attn_train'
            else:
                train_manner = 'hard_train'

        with torch.no_grad():
            if train_manner == 'hard_train':
                _, mask = self.DMIN(fea)
            elif train_manner == 'attn_train':
                attn, mask = self.DMIN(fea, only_attn=True)

        
        if self.args.n_classes == 2:
            mask_add = mask[:, :, 0] + mask[:, :, 1]
        elif self.args.n_classes == 3:
            mask_add = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
        mask_rate = (mask_add != 0).sum() / (mask.shape[0] * mask.shape[1])
        



        if train_manner == 'hard_train':
            discrete_img_scores = self.LIPN(img, phase=train_manner)
        elif train_manner == 'attn_train':
            img_scores, discrete_img_scores = self.LIPN(img, phase=train_manner)


        avg_loss, avg_score_rate = 0, 0
        for i in range(len(discrete_img_scores)):
            if train_manner == 'hard_train':
                score_rate = self.cal_score_rate(discrete_img_scores[i])
                loss = self.train_single_LIPN(discrete_img_scores[i], score_rate, mask, mask_rate)
            elif train_manner == 'attn_train':
                score_rate = self.cal_score_rate(discrete_img_scores[i])
                loss = self.train_single_LIPN(img_scores[i], score_rate, attn, mask_rate)

            avg_loss += loss
            avg_score_rate += score_rate
        
        return avg_loss / len(discrete_img_scores), mask_rate, avg_score_rate / len(discrete_img_scores)

        

    def tune_DMIN(self, fea, label, img=None):    

        with torch.no_grad():
            img_scores = self.LIPN(img, phase='infer')

        bag_logit = self.DMIN(fea, img_scores)[0]
        loss = self.cross_entropy_loss_ls(bag_logit, label)

        return loss
    
    def test_inference(self, fea, label, img, DMIN_model, LIPN_model):
        

        img_scores = LIPN_model(img, phase='infer')

        if self.args.n_classes == 2:
            score_union = (img_scores[:, :, 0] + img_scores[:, :, 1]) > 0
        elif self.args.n_classes == 3:
            score_union = (img_scores[:, :, 0] + img_scores[:, :, 1] + img_scores[:, :, 2]) > 0
        score_rate = score_union.sum() / (img_scores.shape[0] * img_scores.shape[1])

        bag_logit = DMIN_model(fea, img_scores)[0]

        loss = self.cross_entropy_loss(bag_logit, label)
        y_prob = F.softmax(bag_logit, dim=1)
      
        return loss, y_prob, score_rate

    def cal_AUC(self, probs, labels, nclasses):
        '''
            probs(softmaxed): ndarray, [N, nclass] 
            labels(inte number): ndarray, [N, 1] 
        '''
        if nclasses == 2:
            auc_score = roc_auc_score(labels, probs[:, 1])
        else:
            auc_score = roc_auc_score(labels, probs, multi_class='ovr')

        return auc_score
    
    def cal_ACC(self, probs, labels, nclasses):
        '''
            probs(softmaxed): ndarray, [N, nclass] 
            labels(inte number): ndarray, [N, 1] 
        '''
        log = [{"count": 0, "correct": 0} for i in range(nclasses)]
        pred_hat = np.argmax(probs, 1)
        labels = labels.astype(np.int32)

        if nclasses == 2:
            acc_score = accuracy_score(labels, pred_hat)
            precision = precision_score(labels, pred_hat, average='binary')
            recall = recall_score(labels, pred_hat, average='binary')
            f1 = f1_score(labels, pred_hat, average='binary')

            return acc_score, log, precision, recall, f1

        else:
            acc_score = accuracy_score(labels, pred_hat)
            precision = precision_score(labels,pred_hat,average='macro')
            recall = recall_score(labels,pred_hat,average='macro')
            f1 = f1_score(labels, pred_hat, labels=list(range(self.args.n_classes)), average='macro')

            return acc_score, log, precision, recall, f1
