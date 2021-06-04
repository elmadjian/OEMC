import numpy as np
import pandas as pd
import glob
import os
import re

class Scorer():
    
    def __init__(self, base_path=None, model_params=None, folds=None):
        self.base_path = base_path
        self.model = model_params
        self.folds = folds
        self.conf_matrix = [{'tp':0,'tn':0,'fp':0,'fn':0} for i in range(4)]
        self.event_matrix = [{'tp':0, 'tn':0, 'fp':0, 'fn':0} for i in range(4)]
        #self.individual = {}


    def _reset_scores(self):
        self.conf_matrix = [{'tp':0,'tn':0,'fp':0,'fn':0} for i in range(4)]
        self.event_matrix = [{'tp':0, 'tn':0, 'fp':0, 'fn':0} for i in range(4)]


    def score(self):
        for fold in range(self.folds):
            path = self.base_path + '/' + self.model + str(fold+1) + '.npz'
            data = np.load(path, mmap_mode='r')
            preds = data['pred']
            gt = data['gt']
            self._count_sample(preds, gt)
            self._count_event(preds, gt)
        self._show_results_sample()
        self._show_results_event()

    
    def score_ibdt(self, base_path, dataset):
        pattern = os.path.join(base_path, '**/*.csv')
        files = glob.glob(pattern, recursive=True)
        preds = [name for name in files if 'classification.csv' in name]
        gts   = [name for name in files if 'reviewed.csv' in name]
        users = self._index_users(preds, gts, dataset)
        for user in users.keys():
            #self.individual[user] = {'FIX':0, 'SAC':0, 'SP':0}
            self._get_score_user(users[user], user)
            #print(self.individual[user])
        # fix, sac, sp = 0,0,0
        # user_tot = len(self.individual.keys())
        # for user in self.individual.keys():
        #     fix += self.individual[user]['FIX']
        #     sac += self.individual[user]['SAC']
        #     sp  += self.individual[user]['SP']
        # print('FIX F1:', fix/user_tot)
        # print('SAC F1:', sac/user_tot)
        # print('SP F1:', sp/user_tot)
        self._show_results_sample()
        
        

    def _get_score_user(self, user, key):
        for video in user.keys():
            pred = user[video]['pred']
            gt   = user[video]['gt']
            self.score_csv(pred, gt)
            #self._f_score_calc_individual(self.conf_matrix, key)
            #self._reset_scores()
        # user_tot = len(user.keys())
        # self.individual[key]['FIX'] /= user_tot
        # self.individual[key]['SAC'] /= user_tot
        # self.individual[key]['SP'] /= user_tot
        #self._reset_scores()
        



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
        self._count_sample(preds, gt)       


    def _show_results_sample(self):
        print("\n>>> Results on SAMPLE-LEVEL:")
        self._f_score_calc(self.conf_matrix)


    def _show_results_event(self):
        print("\n>>> Results on EVENT-LEVEL:")
        self._f_score_calc(self.event_matrix)


    def _f_score_calc_individual(self, matrix, user):
        for patt in range(len(matrix)):
            tp = matrix[patt]['tp']
            fp = matrix[patt]['fp']
            fn = matrix[patt]['fn']
            precision = 0
            if tp+fp > 0:
                precision = tp/(tp+fp)
            recall = 0
            if tp + fn > 0:
                recall = tp/(tp+fn)
            name = ""
            if patt == 0:
                name = 'FIX'
            elif patt == 1:
                name = 'SAC'
            elif patt == 2:
                name = 'SP'
            else:
                return
            fscore = 0
            if precision * recall != 0:
                fscore = 2*(precision * recall) / (precision + recall)
            self.individual[user][name] += fscore


    def _f_score_calc(self, matrix):
        for patt in range(len(matrix)):
            tp = matrix[patt]['tp']
            fp = matrix[patt]['fp']
            fn = matrix[patt]['fn']
            precision, recall = 0, 0
            if tp + fp > 0:
                precision = tp/(tp+fp)
            if tp + fn > 0:
                recall = tp/(tp+fn)
            self._print_pattern(patt, precision, recall)


    def _print_pattern(self, patt, precision, recall):
        pattern = ""
        if patt == 0:
            pattern = "Fixation"
        elif patt == 1:
            pattern = "Saccade"
        elif patt == 2:
            pattern = "Smooth Pursuit"
        else:
            pattern = "Blink"
        fscore = 0
        if precision * recall != 0:
            fscore = 2*(precision * recall) / (precision + recall)
        print(f'{pattern:14} -> Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {fscore:.4f}')


    def _count_sample(self, preds, gt):
        for i in range(len(preds)):
            p, g = int(preds[i]), int(gt[i])
            vals = {0,1,2,3}
            if p == g:
                self.conf_matrix[p]['tp'] += 1
                vals.remove(p)
            else:
                self.conf_matrix[p]['fp'] += 1
                self.conf_matrix[g]['fn'] += 1
                vals.remove(p)
                vals.remove(g)
            for val in vals:
                self.conf_matrix[val]['tn'] += 1

    
    def _count_event(self, preds, gt, iou=0.5):
        i = 0
        while i < len(gt):
            vals = {0,1,2,3}
            flagged = {0:False, 1:False, 2:False, 3:False}
            g_0 = g_n = int(gt[i])
            match = 0
            ini = i
            while g_0 == g_n and i < len(gt):
                g_n = gt[i]
                if preds[i] == g_n:
                    match += 1            
                flagged[preds[i]] = True
                i += 1
            ratio = 0
            if i != ini:
                ratio = match/(i-ini)
            if ratio > iou:
                self.event_matrix[g_0]['tp'] += 1
            else:
                self.event_matrix[g_0]['fn'] += 1
            vals.remove(g_0)
            for val in vals:
                if flagged[val]:
                    self.event_matrix[val]['fp'] += 1
                else:
                    self.event_matrix[val]['tn'] += 1


if __name__=="__main__":
    scorer = Scorer('outputs/', 'tcn_model_hmr_BATCH-2048_LAYERS-4_EPOCHS-25_FOLD-', 10)
    scorer.score()
    #scorer = Scorer('', '', 1)
    #scorer.score_gazecom('gaze-com-classification')
