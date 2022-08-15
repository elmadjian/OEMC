import numpy as np
import pandas as pd
import glob
import os
import re
import argparse
from sklearn import metrics
import matplotlib.pyplot as plt


class Scorer():
    
    def __init__(self, args):#base_path=None, model_params=None, folds=None):
        self.base_path = args.out
        self.dataset = args.dataset
        self.model = f"{args.model}_model_{args.dataset}_BATCH-{args.batch_size}"
        self.model += f"_EPOCHS-{args.epochs}_FOLD-"
        self.folds = args.folds
        self.sample_preds = []
        self.sample_gt = []
        self.event_preds = []
        self.event_gt = []


    def score(self):
        for fold in range(self.folds):
            path = os.path.join(self.base_path, self.model + str(fold+1) + '.npz')
            data = np.load(path, mmap_mode='r')
            self.sample_preds += data['pred'].tolist()
            self.sample_gt += data['gt'].tolist()
        self._count_event(self.sample_preds, self.sample_gt)
        self.print_results()


    def score_ibdt(self):
        pattern = os.path.join(self.base_path, '**/*.csv')
        files = glob.glob(pattern, recursive=True)
        preds = [name for name in files if 'classification.csv' in name]
        gts   = [name for name in files if 'reviewed.csv' in name]
        users = self._index_users(preds, gts, self.dataset)
        for user in users.keys():
            self._get_score_user(users[user], user)
        self.sample_preds = np.array(self.sample_preds)
        self.sample_gt = np.array(self.sample_gt)
        self.sample_preds = self.sample_preds[self.sample_preds != 3]
        self.sample_gt = self.sample_gt[self.sample_gt != 3]
        self._count_event(self.sample_preds, self.sample_gt)
        self.print_results(ibdt=True)        
        

    def _get_score_user(self, user, key):
        for video in user.keys():
            pred = user[video]['pred']
            gt   = user[video]['gt']
            self.score_csv(pred, gt)


    def _index_users(self, preds, gts, dataset):
        users = {}
        for pred in preds:
            for gt in gts:
                if dataset == 'gazecom':
                    p_vid, p_id = re.findall('\/(\w+)\/(\w+)\/\w+.csv', pred)[0]
                    g_vid, g_id = re.findall('\/(\w+)\/(\w+)\/\w+.csv', gt)[0]
                elif dataset == 'hmr':
                    p_id, p_vid = re.findall('(user_\d+)\/(\d)\/\w+.csv', pred)[0]
                    g_id, g_vid = re.findall('(user_\d+)\/(\d)\/\w+.csv', gt)[0]
                if p_id not in users.keys():
                    users[p_id] = {}
                if p_vid == g_vid and p_id == g_id:
                    users[p_id][p_vid] = {'pred': None, 'gt': None}
                    users[p_id][p_vid]['pred'] = pred
                    users[p_id][p_vid]['gt'] = gt
        return users


    def score_csv(self, output, ground_truth):
        out_df = pd.read_csv(output, names=['i', 'pattern', 'timestamp'])
        gt_df  = pd.read_csv(ground_truth, names=['i', 'pattern', 'timestamp'])
        preds  = out_df['pattern'].tolist()
        gt     = gt_df['pattern'].tolist()
        self.sample_preds += preds
        self.sample_gt += gt      


    def print_results(self, ibdt=False):
        target_names = ['Fixations', 'Saccades', 'Pursuits', 'Noise/Blink']
        if ibdt:
            target_names = target_names[:-1]
        print('SAMPLE-LEVEL metrics\n===================')
        print(metrics.classification_report(self.sample_gt, self.sample_preds, 
                                        target_names=target_names, digits=4))
        metrics.ConfusionMatrixDisplay.from_predictions(self.sample_gt, 
                                self.sample_preds, display_labels=target_names, 
                                cmap='Purples', normalize='pred', values_format='.2f')
        print('EVENT-LEVEL metrics\n===================')
        print(metrics.classification_report(self.event_gt, self.event_preds,
                                        target_names=target_names, digits=4))
        metrics.ConfusionMatrixDisplay.from_predictions(self.event_gt,
                                self.event_preds, display_labels=target_names, 
                                normalize='pred', cmap='Greens', values_format='.2f')
        plt.show()

     
    def _count_event(self, preds, gt):
        i = 0
        while i < len(gt):
            g_0 = g_n = int(gt[i])
            ini, end = i, i
            while g_0 == g_n and i < len(gt):
                g_n = int(gt[i])
                end = i
                i += 1
            if ini == end:
                i += 1
                continue
            pred_event = np.array(preds[ini:end], dtype=int)
            self.event_preds.append(np.bincount(pred_event).argmax())
            self.event_gt.append(g_0)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset',
                        required=True)
    parser.add_argument('-m',
                        '--model',
                        required=False)
    parser.add_argument('-b',
                        '--batch_size',
                        default=2048,
                        required=False,
                        type=int)
    parser.add_argument('-e',
                        '--epochs',
                        default=25,
                        required=False,
                        type=int)
    parser.add_argument('-f',
                        '--folds',
                        default=5,
                        required=False,
                        type=int)
    parser.add_argument('--out',
                        default='outputs/',
                        required=False)
    parser.add_argument('--ibdt',
                        required=False,
                        action='store_true')
    args = parser.parse_args()
    scorer = Scorer(args)
    if args.ibdt:
        scorer.score_ibdt()
    else:
        scorer.score()
