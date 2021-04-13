import numpy as np

class Scorer():
    
    def __init__(self, base_path, model_params, folds):
        self.base_path = base_path
        self.model = model_params
        self.folds = folds
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


    def _show_results_sample(self):
        print("\n>>> Results on SAMPLE-LEVEL:")
        self._f_score_calc(self.conf_matrix)


    def _show_results_event(self):
        print("\n>>> Results on EVENT-LEVEL:")
        self._f_score_calc(self.event_matrix)


    def _f_score_calc(self, matrix):
        for patt in range(len(matrix)):
            tp = matrix[patt]['tp']
            fp = matrix[patt]['fp']
            fn = matrix[patt]['fn']
            precision = tp/(tp+fp)
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
    #scorer = Scorer('outputs/', 'tcn_model_hmr_BATCH-2048_LAYERS-5_EPOCHS-25_FOLD-', 5)
    scorer = Scorer('outputs/', 'tcn_model_gazecom_BATCH-2048_LAYERS-5_EPOCHS-25_FOLD-', 5)
    scorer.score()
