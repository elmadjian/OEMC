import numpy as np
import preprocessor as pr
import argparse
from tcn import TCN
from cnn_lstm import CNN_LSTM
import torch
import os
import scorer


class OnlineSimulator():
    '''
    5. armazenar saída da predição em um vetor 
    6. rodar o scorer no vetor computado
    7. computar o resultado final no scorer depois de todos os folds
    '''

    def __init__(self, args):
        self.args = args
        self.pr = pr.Preprocessor(stride=9, frequency=args.timesteps, offset=0)
        self.conf_matrix = {i:{'tp':0,'tn':0,'fp':0,'fn':0} for i in range(4)}
        self.scorer = scorer.Scorer()


    def simulate(self):
        option = self.args.dataset + '_old'
        fold = self.pr.load_data_k_fold('cached/'+self.pr.append_options(option), 
                                         folds=self.args.folds)
        for fold_i in range(self.args.folds):
            _, _, teX, teY = next(fold)
            features = teX.shape[1]
            window = np.zeros((self.args.timesteps, features))
            model = self._load_model(features, fold_i+1)
            for i, inp in enumerate(teX):
                window = self._fill_up_tensor(inp, window)
                if i > args.timesteps:
                    pred = self._predict(model, window).cpu().numpy()[0][0]
                    gt  = int(teY[i])
                    self._update_conf_matrix(pred, gt)
                    self._show_progress(i, teX)
            print(f'\nFOLD {fold_i}\n------------------')
            self.scorer._f_score_calc(self.conf_matrix)


    def _show_progress(self, i, data):
        if i % 1000 == 0:
            porc = i/len(data) * 100
            print("Progress: {:2.3f}%".format(porc), end='\r', flush=True)
        


    def _load_model(self, features, fold):
        filename = f"{self.args.model_name}_model_{self.args.dataset}_BATCH-"
        filename += f"{self.args.batch_size}_EPOCHS-{self.args.epochs}_FOLD-"
        filename += f"{fold}.pt"
        if self.args.model_name == 'tcn':
            model = TCN(self.args.timesteps, 4, [30]*4,
                kernel_size=self.args.kernel_size, dropout=self.args.dropout)
        else:
            model = CNN_LSTM(self.args.timesteps, 4, self.args.kernel_size,
                self.args.dropout, features, self.args.cnn_layers)
        path = os.path.join('models', filename)
        model.load_state_dict(torch.load(path))
        model.cuda()
        model.eval()
        return model


    def _predict(self, model, sample):
        with torch.no_grad():
            sample = np.array([sample])
            sample = torch.from_numpy(sample).float()
            sample = torch.autograd.Variable(sample, requires_grad=False).cuda()
            pred   = model(sample)
            pred   = pred.data.max(1, keepdim=True)[1]
            return pred

    
    def _fill_up_tensor(self, sample, window):
        window = np.roll(window, -1, axis=0)
        window[-1,:] = sample
        return window


    def _convert_label(self, target):
        if target == 0:
            return 'F'
        if target == 1:
            return 'S'
        if target == 2:
            return 'P'
        if target == 3:
            return 'B'


    def _update_conf_matrix(self, pred, gt):
        patterns = {0,1,2,3}
        if pred == gt:
            patterns.remove(gt)
            self.conf_matrix[gt]['tp'] += 1
            for i in patterns:
                self.conf_matrix[i]['tn'] += 1
        elif pred != gt:
            self.conf_matrix[gt]['fn'] += 1
            patterns.remove(gt)
            self.conf_matrix[pred]['fp'] += 1
            patterns.remove(pred)
            for i in patterns:
                self.conf_matrix[i]['tn'] += 1



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_name',
                        required=True)
    parser.add_argument('-d',
                        '--dataset',
                        required=True)
    parser.add_argument('-b',
                        '--batch_size',
                        required=False,
                        default=2048,
                        type=int)
    parser.add_argument('-e',
                        '--epochs',
                        required=False,
                        default=25,
                        type=int)
    parser.add_argument('-t',
                        '--timesteps',
                        required=True,
                        type=int)
    parser.add_argument('-k',
                        '--kernel_size',
                        required=False,
                        default=5,
                        type=int)
    parser.add_argument('--dropout',
                        required=False,
                        default=0.25,
                        type=float)
    parser.add_argument('--cnn_layers',
                        required=False,
                        default=2,
                        type=int)
    parser.add_argument('-f',
                        '--folds',
                        required=True,
                        default=10,
                        type=int)
    args = parser.parse_args()
    online_sim = OnlineSimulator(args)
    online_sim.simulate()